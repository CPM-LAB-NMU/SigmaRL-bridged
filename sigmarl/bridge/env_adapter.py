"""
Wraps the VMAS/TorchRL environment with a flat float-list API suitable for
serialisation over gRPC/protobuf.  No gRPC imports here — keeps the simulator
logic decoupled from the transport layer.
"""

import numpy as np
import torch
import torch.nn.utils as nn_utils

from torchrl.envs import RewardSum
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import step_mdp

from sigmarl.constants import SCENARIOS
from sigmarl.helper_training import Parameters, TransformedEnvCustom
from sigmarl.modules.decision_making_module import DecisionMakingModule
from sigmarl.scenarios.road_traffic import ScenarioRoadTraffic

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

        # Cache weight count for the default policy architecture
        dm = DecisionMakingModule.from_env(env, params)
        self._n_weights = sum(p.numel() for p in dm.policy.parameters())

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

        self._td[self._env.action_key] = actions
        self._td = self._env.step(self._td)

        next_td = self._td["next"]
        obs = next_td[("agents", "observation")]          # [n_envs, n_agents, obs_dim]
        rewards = next_td[self._env.reward_key].squeeze(-1)  # [n_envs, n_agents]

        done_td = next_td.get("done", next_td.get("terminated"))
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

                done_td = td["next"].get("done", td["next"].get("terminated"))
                done = bool(done_td.any().item()) if done_td is not None else False
                td = step_mdp(td)

            episode_rewards.append(total_reward)

        arr = np.array(episode_rewards)
        return {
            "episode_rewards": episode_rewards,
            "mean_reward": float(arr.mean()),
            "std_reward": float(arr.std()),
        }
