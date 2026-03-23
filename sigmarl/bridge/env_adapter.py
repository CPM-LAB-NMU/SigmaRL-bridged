"""
Wraps the VMAS/TorchRL environment with a flat float-list API suitable for
serialisation over gRPC/protobuf.  No gRPC imports here — keeps the simulator
logic decoupled from the transport layer.
"""

import math
import queue as _queue
import threading

import numpy as np
import torch
import torch.nn.utils as nn_utils

from torchrl.envs import RewardSum
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import step_mdp

from sigmarl.constants import AGENTS, SCENARIOS
from sigmarl.helper_training import Parameters, TransformedEnvCustom
from sigmarl.modules.decision_making_module import DecisionMakingModule
from sigmarl.scenarios.road_traffic import ScenarioRoadTraffic

# ---------------------------------------------------------------------------
# Coordinate transform between the CPM lab frame and the VMAS/SigmaRL frame.
#
# Lab frame   : origin at map corner, x right, y up.
# VMAS frame  : 180° rotation of lab frame, then shifted by (world_x_dim, world_y_dim).
#
# lab → VMAS :  pos_vmas = pos_lab @ R  + shift   (R = [[-1,0],[0,-1]])
# VMAS → lab :  pos_lab  = (pos_vmas - shift) @ R  (R is its own inverse)
#
# Heading:
# lab → VMAS :  heading_vmas = heading_lab + π   (normalised to (-π, π])
# VMAS → lab :  heading_lab  = heading_vmas - π  (normalised to (-π, π])
# ---------------------------------------------------------------------------
_R = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])  # 180° rotation matrix

# ---------------------------------------------------------------------------
# Main-thread render pump.
#
# On macOS, SDL/pygame requires all window operations to happen on the main
# thread.  When the gRPC server calls render from a worker thread it triggers
# a SIGABRT.
#
# server.py sets _RENDER_QUEUE to a queue before starting the server.
# The main thread in serve() drains that queue.  Worker threads post a
# (callable, result_queue) pair and block until the main thread runs it.
# ---------------------------------------------------------------------------
_RENDER_QUEUE: "_queue.Queue | None" = None

_DEFAULT_CONFIG = {
    "scenario_type": "intersection_1",
    "n_agents": 4,
    "n_envs": 1,
    "max_steps": 128,
    "device": "cpu",
    "random_seed": 0,
    "is_partial_observation": True,
    "is_ego_view": True,
}


class EnvAdapter:
    """
    Stateful wrapper around a single VMAS environment instance.

    Thread-safety: not thread-safe.  The gRPC server serialises calls through
    a single servicer instance, so this is fine for the default use-case.
    """

    def __init__(self, config: dict | None = None):
        self._env: TransformedEnvCustom | None = None
        self._td = None        # current TensorDict (mutated each step)
        self._params: Parameters | None = None
        self._n_weights: int | None = None

        # Shape cache
        self._obs_dim: int = 0
        self._action_dim: int = 0
        self._n_agents: int = 0
        self._n_envs: int = 0
        self._action_low: list[float] = []
        self._action_high: list[float] = []

        self._shift: torch.Tensor | None = None  # set in _build from scenario dims
        self._render_mode: str = ""              # "" | "human" | "rgb_array"
        self._frame_list: list = []
        self._build(config or _DEFAULT_CONFIG)

    # ------------------------------------------------------------------ build

    def _build(self, config: dict) -> None:
        cfg = {**_DEFAULT_CONFIG, **{k: v for k, v in config.items() if v}}

        scenario_type = cfg["scenario_type"]
        n_agents = cfg["n_agents"]
        if scenario_type in SCENARIOS:
            n_agents = SCENARIOS[scenario_type].get("n_agents", n_agents)

        params = Parameters(
            scenario_type=scenario_type,
            n_agents=n_agents,
            num_vmas_envs=cfg["n_envs"],
            max_steps=cfg["max_steps"],
            device=cfg["device"],
            random_seed=cfg["random_seed"],
            is_partial_observation=cfg["is_partial_observation"],
            is_ego_view=cfg["is_ego_view"],
        )

        scenario = ScenarioRoadTraffic()
        scenario.parameters = params

        env = VmasEnv(
            scenario=scenario,
            num_envs=params.num_vmas_envs,
            continuous_actions=True,
            max_steps=params.max_steps,
            device=params.device,
            n_agents=params.n_agents,
        )
        env = TransformedEnvCustom(
            env,
            RewardSum(
                in_keys=[env.reward_key],
                out_keys=[("agents", "episode_reward")],
            ),
        )

        self._env = env
        self._params = params

        # Cache shapes
        obs_spec = env.observation_spec[("agents", "observation")]
        act_spec = env.unbatched_action_spec[env.action_key]
        self._obs_dim = int(obs_spec.shape[-1])
        self._action_dim = int(act_spec.shape[-1])
        self._n_agents = params.n_agents
        self._n_envs = params.num_vmas_envs
        self._action_low = act_spec.space.low.cpu().numpy().flatten().tolist()
        self._action_high = act_spec.space.high.cpu().numpy().flatten().tolist()

        # Coordinate transform shift vector (world dimensions from scenario config)
        scenario_meta = SCENARIOS.get(params.scenario_type, {})
        world_x = float(scenario_meta.get("world_x_dim", 4.5))
        world_y = float(scenario_meta.get("world_y_dim", 4.0))
        self._shift = torch.tensor([[world_x, world_y]], dtype=torch.float32)

        # Cache weight count for the default policy architecture
        dm = DecisionMakingModule.from_env(env, params)
        self._n_weights = sum(p.numel() for p in dm.policy.parameters())

    # ------------------------------------------------------------------ render

    def set_render_mode(self, mode: str) -> None:
        """Set rendering mode: '' = off, 'human' = live window, 'rgb_array' = collect frames."""
        self._render_mode = mode
        self._frame_list = []

    def _render_step(self) -> None:
        if not self._render_mode:
            return
        if _RENDER_QUEUE is not None:
            # Running inside the gRPC server: delegate to main thread to avoid
            # SDL/pygame SIGABRT on macOS when called from a worker thread.
            result_q: _queue.Queue = _queue.Queue()
            _RENDER_QUEUE.put((self._do_render, result_q))
            frame = result_q.get()  # blocks until main thread completes the call
        else:
            frame = self._do_render()
        if self._render_mode == "rgb_array" and frame is not None:
            self._frame_list.append(frame)

    def _do_render(self):
        return self._env.render(
            mode=self._render_mode,
            visualize_when_rgb=(self._render_mode == "human"),
        )

    def save_video(self, path: str) -> None:
        """Flush collected rgb_array frames to an MP4 file at 20 fps (dt=0.05s)."""
        from vmas.simulator.utils import save_video as vmas_save_video
        vmas_save_video(path, self._frame_list, fps=20)
        self._frame_list = []

    # ------------------------------------------------------------------ spaces

    @property
    def spaces(self) -> dict:
        return {
            "n_agents": self._n_agents,
            "n_envs": self._n_envs,
            "obs_dim": self._obs_dim,
            "action_dim": self._action_dim,
            "action_low": self._action_low,
            "action_high": self._action_high,
        }

    @property
    def weight_count(self) -> int:
        return self._n_weights

    # ------------------------------------------------------------------ reset

    def reset(self, seed: int = 0) -> dict:
        if seed:
            torch.manual_seed(seed)
        self._td = self._env.reset()
        obs = self._td[("agents", "observation")]  # [n_envs, n_agents, obs_dim]
        return {
            "observations": obs.cpu().numpy().flatten().tolist(),
            "rewards": [0.0] * (self._n_envs * self._n_agents),
            "dones": [False] * self._n_envs,
            "n_envs": self._n_envs,
            "n_agents": self._n_agents,
            "obs_dim": self._obs_dim,
        }

    # ------------------------------------------------------------------ step

    def step(self, actions_flat: list[float]) -> dict:
        actions = torch.tensor(
            actions_flat, dtype=torch.float32, device=self._params.device
        ).reshape(self._n_envs, self._n_agents, self._action_dim)

        # Clip to valid physical range so out-of-bound values from any client
        # don't crash VMAS (e.g. a Java network outputting raw tanh values).
        low  = torch.tensor(self._action_low,  device=self._params.device).reshape(1, self._n_agents, self._action_dim)
        high = torch.tensor(self._action_high, device=self._params.device).reshape(1, self._n_agents, self._action_dim)
        actions = torch.clamp(actions, low, high)

        self._td[self._env.action_key] = actions
        self._td = self._env.step(self._td)

        next_td = self._td["next"]
        obs = next_td[("agents", "observation")]          # [n_envs, n_agents, obs_dim]
        rewards = next_td[self._env.reward_key].squeeze(-1)  # [n_envs, n_agents]

        done_td = next_td.get("done")
        if done_td is None:
            done_td = next_td.get("terminated")
        if done_td is None:
            done_td = next_td.get("truncated")
        if done_td is not None:
            dones = done_td.squeeze(-1).flatten().tolist()
        else:
            dones = [False] * self._n_envs

        self._td = step_mdp(self._td)

        return {
            "observations": obs.cpu().numpy().flatten().tolist(),
            "rewards": rewards.cpu().numpy().flatten().tolist(),
            "dones": dones if isinstance(dones, list) else [dones],
            "n_envs": self._n_envs,
            "n_agents": self._n_agents,
            "obs_dim": self._obs_dim,
        }

    # ------------------------------------------------------------------ physical interface

    def _vmas_to_lab_pos(self, pos_vmas: torch.Tensor) -> torch.Tensor:
        """Convert a [*, 2] position tensor from VMAS frame to lab frame."""
        return (pos_vmas - self._shift.to(pos_vmas.device)) @ _R.to(pos_vmas.device)

    def _vmas_to_lab_heading(self, heading_vmas: torch.Tensor) -> torch.Tensor:
        """Convert a [*] heading tensor from VMAS frame to lab frame (radians)."""
        h = heading_vmas - math.pi
        # normalise to (-π, π]
        return torch.where(h <= -math.pi, h + 2 * math.pi, h)

    def _agent_physical_states(self) -> list[dict]:
        """
        Extract per-agent physical state from the live VMAS world and return
        it as a list of dicts in the lab coordinate frame.

        Returns one dict per (env, agent) pair, with keys:
            vehicle_id, x, y, heading, speed
        """
        world = self._env.base_env.scenario.world
        states = []
        for agent_idx, agent in enumerate(world.agents):
            pos_lab  = self._vmas_to_lab_pos(agent.state.pos)   # [n_envs, 2]
            rot_vmas = agent.state.rot.squeeze(-1)               # [n_envs]
            head_lab = self._vmas_to_lab_heading(rot_vmas)       # [n_envs]
            speed    = agent.state.vel.norm(dim=-1)              # [n_envs]

            for env_idx in range(self._n_envs):
                states.append({
                    "vehicle_id": agent_idx,
                    "env_idx":    env_idx,
                    "x":          float(pos_lab[env_idx, 0].item()),
                    "y":          float(pos_lab[env_idx, 1].item()),
                    "heading":    float(head_lab[env_idx].item()),
                    "speed":      float(speed[env_idx].item()),
                })
        return states

    def reset_physical(self, seed: int = 0) -> dict:
        """
        Reset and return per-agent physical state in the lab coordinate frame.
        Designed for n_envs == 1.
        """
        if seed:
            torch.manual_seed(seed)
        self._td = self._env.reset()

        done_td = self._td.get("done", self._td.get("terminated"))
        dones = done_td.squeeze(-1).flatten().tolist() if done_td is not None else [False] * self._n_envs

        self._render_step()
        return {
            "states": self._agent_physical_states(),
            "dones":  dones if isinstance(dones, list) else [dones],
            "n_envs": self._n_envs,
        }

    def step_physical(self, commands: list[dict]) -> dict:
        """
        Advance the simulation one step from per-agent speed+curvature commands
        (lab interface format) and return the resulting physical state.

        Each command dict must have keys: vehicle_id, speed, curvature.
        curvature [1/m]: positive = left turn.
        Designed for n_envs == 1.
        """
        l_wb = AGENTS["l_wb"]

        # Build action tensor [n_envs, n_agents, 2] = [speed, steering_angle]
        # curvature → steering via kinematic bicycle model: steer = atan(curv * l_wb)
        actions = torch.zeros(
            self._n_envs, self._n_agents, self._action_dim,
            dtype=torch.float32, device=self._params.device,
        )
        for cmd in commands:
            a_idx = int(cmd["vehicle_id"])
            if a_idx >= self._n_agents:
                continue
            speed    = float(cmd["speed"])
            steering = math.atan(float(cmd["curvature"]) * l_wb)
            actions[:, a_idx, 0] = speed
            actions[:, a_idx, 1] = steering

        # Clip to physical limits
        low  = torch.tensor(self._action_low,  device=self._params.device).reshape(1, self._n_agents, self._action_dim)
        high = torch.tensor(self._action_high, device=self._params.device).reshape(1, self._n_agents, self._action_dim)
        actions = torch.clamp(actions, low, high)

        self._td[self._env.action_key] = actions
        self._td = self._env.step(self._td)

        next_td = self._td["next"]
        done_td = next_td.get("done")
        if done_td is None:
            done_td = next_td.get("terminated")
        if done_td is None:
            done_td = next_td.get("truncated")
        dones = done_td.squeeze(-1).flatten().tolist() if done_td is not None else [False] * self._n_envs

        self._td = step_mdp(self._td)

        self._render_step()
        return {
            "states": self._agent_physical_states(),
            "dones":  dones if isinstance(dones, list) else [dones],
            "n_envs": self._n_envs,
        }

    # ------------------------------------------------------------------ EA

    def evaluate_weights(self, weights_flat: list[float], n_episodes: int) -> dict:
        """
        Load a flat weight vector into the default policy architecture, run
        n_episodes full episodes, and return per-episode cumulative rewards.

        Java owns the evolutionary loop; this method is the fitness oracle.
        """
        dm = DecisionMakingModule.from_env(self._env, self._params)
        weights_tensor = torch.tensor(weights_flat, dtype=torch.float32)
        nn_utils.vector_to_parameters(weights_tensor, dm.policy.parameters())
        dm.policy.eval()

        episode_rewards: list[float] = []

        for _ in range(n_episodes):
            td = self._env.reset()
            total_reward = 0.0
            done = False

            while not done:
                with torch.no_grad():
                    td = dm.policy(td)
                td = self._env.step(td)

                reward = td["next"][self._env.reward_key].sum().item()
                total_reward += reward

                done_td = td["next"].get("done")
                if done_td is None:
                    done_td = td["next"].get("terminated")
                if done_td is None:
                    done_td = td["next"].get("truncated")
                done = bool(done_td.any().item()) if done_td is not None else False
                td = step_mdp(td)

            episode_rewards.append(total_reward)

        arr = np.array(episode_rewards)
        return {
            "episode_rewards": episode_rewards,
            "mean_reward": float(arr.mean()),
            "std_reward": float(arr.std()),
        }
