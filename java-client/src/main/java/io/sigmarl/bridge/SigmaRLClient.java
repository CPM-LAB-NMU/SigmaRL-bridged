package io.sigmarl.bridge;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Thin wrapper around the generated gRPC stub.
 *
 * <p>Hides channel management and provides a plain Java API matching the four
 * core operations:
 * <ol>
 *   <li>{@link #getSpaces()} — introspect observation/action shapes</li>
 *   <li>{@link #configure(ScenarioConfig)} — (re-)configure the scenario</li>
 *   <li>{@link #reset()} — start a new episode</li>
 *   <li>{@link #step(float[])} — advance one timestep</li>
 *   <li>{@link #evaluateWeights(float[], int)} — EA fitness oracle</li>
 * </ol>
 *
 * <p>Usage:
 * <pre>{@code
 * try (SigmaRLClient client = new SigmaRLClient("localhost", 50051)) {
 *     SpacesInfo spaces = client.getSpaces();
 *     client.reset();
 *     StepResult result = client.step(new float[]{0.1f, 0.0f, 0.1f, 0.0f, ...});
 * }
 * }</pre>
 */
public class SigmaRLClient implements AutoCloseable {

    private final ManagedChannel channel;
    private final SigmaRLEnvGrpc.SigmaRLEnvBlockingStub stub;

    public SigmaRLClient(String host, int port) {
        this.channel = ManagedChannelBuilder
                .forAddress(host, port)
                .usePlaintext()
                .build();
        this.stub = SigmaRLEnvGrpc.newBlockingStub(channel);
    }

    // ---------------------------------------------------------------- spaces

    public SpacesInfo getSpaces() {
        return stub.getSpaces(Empty.getDefaultInstance());
    }

    public long getWeightCount() {
        return stub.getWeightCount(Empty.getDefaultInstance()).getNWeights();
    }

    // ---------------------------------------------------------------- configure

    /** Configure (or reconfigure) the scenario before the first Reset. */
    public Ack configure(ScenarioConfig config) {
        return stub.configure(config);
    }

    /** Convenience builder for the most common config fields. */
    public static ScenarioConfig scenarioConfig(
            String scenarioType, int nAgents, int nEnvs, int maxSteps, String device) {
        return ScenarioConfig.newBuilder()
                .setScenarioType(scenarioType)
                .setNAgents(nAgents)
                .setNEnvs(nEnvs)
                .setMaxSteps(maxSteps)
                .setDevice(device)
                .build();
    }

    // ---------------------------------------------------------------- reset

    public StepResponse reset() {
        return stub.reset(ResetRequest.getDefaultInstance());
    }

    public StepResponse reset(int seed) {
        return stub.reset(ResetRequest.newBuilder().setSeed(seed).build());
    }

    // ---------------------------------------------------------------- step

    /**
     * Advance the environment one timestep.
     *
     * @param actions flat row-major array of shape [n_envs * n_agents * action_dim]
     */
    public StepResponse step(float[] actions) {
        StepRequest.Builder req = StepRequest.newBuilder();
        for (float a : actions) req.addActions(a);
        return stub.step(req.build());
    }

    // ---------------------------------------------------------------- EA

    /**
     * Evaluate a flat policy weight vector over {@code nEpisodes} episodes.
     *
     * @param weights    flat array of length {@link #getWeightCount()}
     * @param nEpisodes  number of episodes to average over
     */
    public EvaluateResponse evaluateWeights(float[] weights, int nEpisodes) {
        EvaluateRequest.Builder req = EvaluateRequest.newBuilder()
                .setNEpisodes(nEpisodes);
        for (float w : weights) req.addWeights(w);
        return stub.evaluateWeights(req.build());
    }

    // ---------------------------------------------------------------- helpers

    /**
     * Extract the observation for a specific (env, agent) pair from a flat
     * StepResponse observation array.
     */
    public static float[] agentObs(StepResponse resp, int envIdx, int agentIdx) {
        int obsDim  = resp.getObsDim();
        int nAgents = resp.getNAgents();
        int offset  = (envIdx * nAgents + agentIdx) * obsDim;
        List<Float> all = resp.getObservationsList();
        float[] obs = new float[obsDim];
        for (int i = 0; i < obsDim; i++) obs[i] = all.get(offset + i);
        return obs;
    }

    /**
     * Extract the reward for a specific (env, agent) pair from a flat
     * StepResponse rewards array.
     */
    public static float agentReward(StepResponse resp, int envIdx, int agentIdx) {
        return resp.getRewards(envIdx * resp.getNAgents() + agentIdx);
    }

    /** True if any environment in the batch is done. */
    public static boolean anyDone(StepResponse resp) {
        for (boolean d : resp.getDonesList()) if (d) return true;
        return false;
    }

    // ---------------------------------------------------------------- lifecycle

    @Override
    public void close() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }
}
