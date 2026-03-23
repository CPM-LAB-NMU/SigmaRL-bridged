"""
gRPC server — exposes SigmaRL/VMAS as a language-agnostic training backend.

Start with:
    python -m sigmarl.bridge.server          # default port 50051
    python -m sigmarl.bridge.server 50052    # custom port
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from concurrent import futures

import grpc

# Generated stubs (run scripts/generate_proto.sh first)
from sigmarl.bridge import sigmarl_env_pb2, sigmarl_env_pb2_grpc
import sigmarl.bridge.env_adapter as _adapter_mod
from sigmarl.bridge.env_adapter import EnvAdapter

log = logging.getLogger(__name__)


def _config_from_proto(proto_cfg) -> dict:
    """Convert a ScenarioConfig proto message to a plain dict."""
    return {
        "scenario_type": proto_cfg.scenario_type or None,
        "n_agents":      proto_cfg.n_agents or None,
        "n_envs":        proto_cfg.n_envs or None,
        "max_steps":     proto_cfg.max_steps or None,
        "device":        proto_cfg.device or None,
        "random_seed":   proto_cfg.random_seed,
        "is_partial_observation": proto_cfg.is_partial_observation,
        "is_ego_view":            proto_cfg.is_ego_view,
        "render_mode":   proto_cfg.render_mode or "",
    }


class SigmaRLServicer(sigmarl_env_pb2_grpc.SigmaRLEnvServicer):

    def __init__(self):
        self._adapter = EnvAdapter()  # initialised with defaults
        log.info("SigmaRL servicer ready (default config)")

    # ------------------------------------------------------------ introspect

    def GetSpaces(self, request, context):
        s = self._adapter.spaces
        return sigmarl_env_pb2.SpacesInfo(
            n_agents=s["n_agents"],
            n_envs=s["n_envs"],
            obs_dim=s["obs_dim"],
            action_dim=s["action_dim"],
            action_low=s["action_low"],
            action_high=s["action_high"],
        )

    def GetWeightCount(self, request, context):
        return sigmarl_env_pb2.WeightCountInfo(
            n_weights=self._adapter.weight_count
        )

    # ------------------------------------------------------------ configure

    def Configure(self, request, context):
        try:
            cfg = _config_from_proto(request)
            self._adapter = EnvAdapter(cfg)
            if cfg["render_mode"]:
                self._adapter.set_render_mode(cfg["render_mode"])
                log.info("Rendering enabled: mode=%s", cfg["render_mode"])
            return sigmarl_env_pb2.Ack(success=True, message="Environment configured")
        except Exception as exc:
            log.exception("Configure failed")
            return sigmarl_env_pb2.Ack(success=False, message=str(exc))

    # ------------------------------------------------------------ reset / step (ML interface)

    def Reset(self, request, context):
        r = self._adapter.reset(seed=request.seed)
        return sigmarl_env_pb2.StepResponse(**r)

    def Step(self, request, context):
        r = self._adapter.step(list(request.actions))
        return sigmarl_env_pb2.StepResponse(**r)

    # ------------------------------------------------------------ video / render

    def SetRenderMode(self, request, context):
        try:
            self._adapter.set_render_mode(request.mode)
            msg = f"Render mode set to '{request.mode}'"
            if request.mode:
                log.info(msg)
            return sigmarl_env_pb2.Ack(success=True, message=msg)
        except Exception as exc:
            log.exception("SetRenderMode failed")
            return sigmarl_env_pb2.Ack(success=False, message=str(exc))

    def SaveVideo(self, request, context):
        try:
            self._adapter.save_video(request.path)
            return sigmarl_env_pb2.Ack(success=True, message=f"Video saved to {request.path}.mp4")
        except Exception as exc:
            log.exception("SaveVideo failed")
            return sigmarl_env_pb2.Ack(success=False, message=str(exc))

    # ------------------------------------------------------------ physical interface

    def ResetPhysical(self, request, context):
        r = self._adapter.reset_physical(seed=request.seed)
        return sigmarl_env_pb2.PhysicalStepResponse(
            states=[sigmarl_env_pb2.VehicleStateMsg(**{k: v for k, v in s.items() if k != "env_idx"})
                    for s in r["states"]],
            dones=r["dones"],
            n_envs=r["n_envs"],
        )

    def StepPhysical(self, request, context):
        commands = [
            {"vehicle_id": cmd.vehicle_id, "speed": cmd.speed, "curvature": cmd.curvature}
            for cmd in request.commands
        ]
        r = self._adapter.step_physical(commands)
        return sigmarl_env_pb2.PhysicalStepResponse(
            states=[sigmarl_env_pb2.VehicleStateMsg(**{k: v for k, v in s.items() if k != "env_idx"})
                    for s in r["states"]],
            dones=r["dones"],
            n_envs=r["n_envs"],
        )

    # ------------------------------------------------------------ EA

    def EvaluateWeights(self, request, context):
        # Optional per-call config override
        if request.config.scenario_type:
            try:
                self._adapter = EnvAdapter(_config_from_proto(request.config))
            except Exception as exc:
                log.exception("EvaluateWeights: Configure failed")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        r = self._adapter.evaluate_weights(
            weights_flat=list(request.weights),
            n_episodes=request.n_episodes or 1,
        )
        return sigmarl_env_pb2.EvaluateResponse(**r)


def serve(port: int = 50051, max_workers: int = 1) -> None:
    # Install the main-thread render queue so worker threads delegate SDL/pygame
    # calls here instead of crashing on macOS (SDL requires the main thread).
    render_queue: queue.Queue = queue.Queue()
    _adapter_mod._RENDER_QUEUE = render_queue

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    sigmarl_env_pb2_grpc.add_SigmaRLEnvServicer_to_server(SigmaRLServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("SigmaRL gRPC server listening on port %d", port)

    # Signal the pump loop when the server shuts down
    stop_event = threading.Event()
    def _watch_server():
        server.wait_for_termination()
        stop_event.set()
    threading.Thread(target=_watch_server, daemon=True).start()

    # Main thread render pump — drains render jobs posted by gRPC workers
    log.info("Main-thread render pump active")
    while not stop_event.is_set():
        try:
            render_fn, result_q = render_queue.get(timeout=0.05)
            result_q.put(render_fn())
        except queue.Empty:
            pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    serve(port=port)
