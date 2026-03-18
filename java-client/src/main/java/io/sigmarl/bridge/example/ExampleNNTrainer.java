package io.sigmarl.bridge.example;

import io.sigmarl.bridge.SigmaRLClient;
import io.sigmarl.bridge.ScenarioConfig;
import io.sigmarl.bridge.StepResponse;

import java.util.List;
import java.util.Random;

/**
 * Skeleton for gradient-based neural network training with SigmaRL as the env.
 *
 * <p>Java owns the network definition and the learning algorithm.  Python handles
 * environment simulation, observation computation, and reward shaping.
 *
 * <p>This example shows the step-by-step interaction loop.  You would replace the
 * random policy with your own forward pass, and the placeholder gradient update
 * with real backpropagation (e.g. via DL4J / ND4J).
 *
 * <p>Run after starting the Python server:
 * <pre>
 *   python -m sigmarl.bridge.server
 *   java -cp target/sigmarl-java-client-*.jar io.sigmarl.bridge.example.ExampleNNTrainer
 * </pre>
 */
public class ExampleNNTrainer {

    private static final int EPISODES    = 100;
    private static final int LOG_EVERY   = 10;

    public static void main(String[] args) throws Exception {
        String host = args.length > 0 ? args[0] : "localhost";
        int    port = args.length > 1 ? Integer.parseInt(args[1]) : 50051;

        try (SigmaRLClient client = new SigmaRLClient(host, port)) {

            // Configure scenario — adjust to match your network's input size
            ScenarioConfig cfg = SigmaRLClient.scenarioConfig(
                    "intersection_1", 4, /*n_envs=*/1, /*max_steps=*/128, "cpu");
            client.configure(cfg);

            var spaces = client.getSpaces();
            int nAgents   = spaces.getNAgents();
            int obsDim    = spaces.getObsDim();
            int actionDim = spaces.getActionDim();
            System.out.printf("n_agents=%d  obs_dim=%d  action_dim=%d%n",
                    nAgents, obsDim, actionDim);

            // ── Replace this with your network initialisation ──────────────
            SimpleNetwork net = new SimpleNetwork(obsDim, actionDim, 64, new Random(0));

            float totalEpisodeReward = 0f;

            for (int ep = 0; ep < EPISODES; ep++) {
                StepResponse state = client.reset();
                float episodeReward = 0f;
                boolean done = false;

                while (!done) {
                    // Collect actions for all agents in env 0
                    float[] actions = new float[nAgents * actionDim];
                    for (int a = 0; a < nAgents; a++) {
                        float[] obs    = SigmaRLClient.agentObs(state, /*env=*/0, a);
                        float[] action = net.forward(obs);
                        System.arraycopy(action, 0, actions, a * actionDim, actionDim);
                    }

                    StepResponse next = client.step(actions);

                    // Accumulate reward (sum over agents)
                    for (int a = 0; a < nAgents; a++) {
                        episodeReward += SigmaRLClient.agentReward(next, 0, a);
                    }

                    // ── Replace with your backprop / policy-gradient update ─
                    // e.g.:
                    //   float loss = policyGradientLoss(state, actions, rewards);
                    //   net.backward(loss);
                    //   net.updateWeights(learningRate);

                    done  = SigmaRLClient.anyDone(next);
                    state = next;
                }

                totalEpisodeReward += episodeReward;
                if ((ep + 1) % LOG_EVERY == 0) {
                    System.out.printf("Episode %4d | mean_reward=%.3f%n",
                            ep + 1, totalEpisodeReward / LOG_EVERY);
                    totalEpisodeReward = 0f;
                }
            }
        }
    }

    // ── Placeholder single-hidden-layer network ────────────────────────────
    // Replace with DL4J MultiLayerNetwork, your own autograd library, etc.

    static class SimpleNetwork {
        private final float[][] w1, w2;
        private final float[]   b1, b2;
        private final int       hiddenDim;
        private final Random    rng;

        SimpleNetwork(int inputDim, int outputDim, int hiddenDim, Random rng) {
            this.hiddenDim = hiddenDim;
            this.rng = rng;
            w1 = randn(hiddenDim, inputDim,  rng, 0.1f);
            b1 = new float[hiddenDim];
            w2 = randn(outputDim, hiddenDim, rng, 0.1f);
            b2 = new float[outputDim];
        }

        float[] forward(float[] obs) {
            float[] hidden = new float[hiddenDim];
            for (int h = 0; h < hiddenDim; h++) {
                float sum = b1[h];
                for (int i = 0; i < obs.length; i++) sum += w1[h][i] * obs[i];
                hidden[h] = (float) Math.tanh(sum);
            }
            float[] out = new float[w2.length];
            for (int o = 0; o < out.length; o++) {
                float sum = b2[o];
                for (int h = 0; h < hiddenDim; h++) sum += w2[o][h] * hidden[h];
                out[o] = (float) Math.tanh(sum);  // bounded actions
            }
            return out;
        }

        private static float[][] randn(int rows, int cols, Random rng, float scale) {
            float[][] m = new float[rows][cols];
            for (float[] row : m)
                for (int j = 0; j < cols; j++)
                    row[j] = (float) rng.nextGaussian() * scale;
            return m;
        }
    }
}
