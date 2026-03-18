"""
gRPC server — exposes SigmaRL/VMAS as a language-agnostic training backend.

Start with:
    python -m sigmarl.bridge.server          # default port 50051
    python -m sigmarl.bridge.server 50052    # custom port
"""

from __future__ import annotations

import logging
import sys
from concurrent import futures

import grpc

# Generated stubs (run scripts/generate_proto.sh first)
from sigmarl.bridge import sigmarl_env_pb2, sigmarl_env_pb2_grpc
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
            self._adapter = EnvAdapter(_config_from_proto(request))
            return sigmarl_env_pb2.Ack(success=True, message="Environment configured")
        except Exception as exc:
            log.exception("Configure failed")
            return sigmarl_env_pb2.Ack(success=False, message=str(exc))

    # ------------------------------------------------------------ reset / step

    def Reset(self, request, context):
        r = self._adapter.reset(seed=request.seed)
        return sigmarl_env_pb2.StepResponse(**r)

    def Step(self, request, context):
        r = self._adapter.step(list(request.actions))
        return sigmarl_env_pb2.StepResponse(**r)

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


def serve(port: int = 50051, max_workers: int = 4) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    sigmarl_env_pb2_grpc.add_SigmaRLEnvServicer_to_server(SigmaRLServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("SigmaRL gRPC server listening on port %d", port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    serve(port=port)
