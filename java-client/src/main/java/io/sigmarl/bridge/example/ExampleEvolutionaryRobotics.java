package io.sigmarl.bridge.example;

import io.sigmarl.bridge.ScenarioConfig;
import io.sigmarl.bridge.SigmaRLClient;
import io.sigmarl.bridge.SpacesInfo;
import io.sigmarl.bridge.StepResponse;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Evolutionary Robotics demo using SigmaRL's <b>ML observation interface</b>.
 *
 * <p>Students implement every component from scratch:
 * <ol>
 *   <li><b>Neural network</b> — maps the simulator's rich observation vector
 *       (path deviation, heading error, neighbour positions, look-ahead path,
 *       …) to a driving command ({@code speed, steering_angle}).  The
 *       observation is deliberately opaque: students treat the environment as
 *       a black box, which is the standard RL setup.</li>
 *   <li><b>Evolutionary algorithm</b> — (μ, λ)-ES with Gaussian mutation.</li>
 *   <li><b>Fitness function</b> — cumulative reward returned by the simulator.
 *       Students can replace this with any scalar derived from
 *       {@link SigmaRLClient#agentReward} or step count.</li>
 * </ol>
 *
 * <p>The Python server acts as a fully self-contained physics + reward oracle.
 * Java owns the entire learning loop.
 *
 * <p>Run with the Python server active:
 * <pre>
 *   python -m sigmarl.bridge.server
 *   java -cp target/sigmarl-java-client-*.jar \
 *        io.sigmarl.bridge.example.ExampleEvolutionaryRobotics
 *   # with live rendering during the final replay:
 *   java -cp target/sigmarl-java-client-*.jar \
 *        io.sigmarl.bridge.example.ExampleEvolutionaryRobotics --render
 * </pre>
 */
public class ExampleEvolutionaryRobotics {

    // ── ES hyper-parameters ───────────────────────────────────────────────
    private static final int    POPULATION   = 20;   // λ — offspring per generation
    private static final int    SURVIVORS    = 5;    // μ — parents selected each gen
    private static final int    GENERATIONS  = 50;
    private static final double MUTATION_STD = 0.05; // Gaussian noise σ
    private static final int    HIDDEN_DIM   = 64;
    private static final int    LOG_EVERY    = 5;

    public static void main(String[] args) throws Exception {
        // Usage: [host] [port] [--render | --save-video PATH]
        //   --render          open a live pygame window while the best individual runs
        //   --save-video PATH save a video of the best individual to PATH.mp4
        String host       = "localhost";
        int    port       = 50051;
        String renderMode = "";       // "" | "human" | "rgb_array"
        String videoPath  = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--render":      renderMode = "human";      break;
                case "--save-video":  renderMode = "rgb_array";
                                      videoPath  = args[++i];   break;
                default:
                    if (i == 0) host = args[i];
                    else if (i == 1) port = Integer.parseInt(args[i]);
            }
        }

        try (SigmaRLClient client = new SigmaRLClient(host, port)) {

            // Configure scenario — no render_mode here so training runs at full speed.
            ScenarioConfig cfg = SigmaRLClient.scenarioConfig(
                    "intersection_1", 4, /*n_envs=*/1, /*max_steps=*/128, "cpu");
            client.configure(cfg);

            SpacesInfo spaces    = client.getSpaces();
            int   nAgents   = spaces.getNAgents();
            int   nEnvs     = spaces.getNEnvs();
            int   obsDim    = spaces.getObsDim();
            int   actionDim = spaces.getActionDim();

            // Action bounds from the server — used to scale NN output correctly.
            float[] actionLow  = toFloatArray(spaces.getActionLowList());
            float[] actionHigh = toFloatArray(spaces.getActionHighList());

            System.out.printf(
                    "Environment: %d agents | obs_dim=%d | action_dim=%d%n",
                    nAgents, obsDim, actionDim);
            System.out.printf(
                    "Network per agent: %d -> %d -> %d%n%n",
                    obsDim, HIDDEN_DIM, actionDim);

            printObservationGuide(obsDim);

            Random rng = new Random(42);
            Network net = new Network(obsDim, actionDim, HIDDEN_DIM, actionLow, actionHigh);

            int       nWeights   = net.weightCount();
            float[][] population = randomPopulation(POPULATION, nWeights, rng);
            float[]   fitness    = new float[POPULATION];

            for (int gen = 0; gen < GENERATIONS; gen++) {

                // ── evaluate each individual ─────────────────────────────
                for (int i = 0; i < POPULATION; i++) {
                    net.loadWeights(population[i]);
                    fitness[i] = runEpisode(client, net, nAgents, nEnvs, actionDim);
                }

                // ── report ───────────────────────────────────────────────
                int[] ranking = argsort(fitness);
                float best = fitness[ranking[POPULATION - 1]];
                float mean = mean(fitness);

                if ((gen + 1) % LOG_EVERY == 0 || gen == 0) {
                    System.out.printf("Gen %3d/%d | best=%.3f  mean=%.3f%n",
                            gen + 1, GENERATIONS, best, mean);
                }

                // ── (μ, λ)-selection: top-μ become parents ───────────────
                float[][] parents = new float[SURVIVORS][nWeights];
                for (int k = 0; k < SURVIVORS; k++) {
                    parents[k] = population[ranking[POPULATION - 1 - k]].clone();
                }

                // ── generate next population by mutation ─────────────────
                float[][] next = new float[POPULATION][nWeights];
                for (int i = 0; i < POPULATION; i++) {
                    float[] parent = parents[rng.nextInt(SURVIVORS)];
                    next[i] = mutate(parent, MUTATION_STD, rng);
                }
                population = next;
            }

            System.out.println("\nEvolution complete.");

            // ── optional: replay best individual with rendering ───────────
            if (!renderMode.isEmpty()) {
                int[] ranking = argsort(fitness);
                net.loadWeights(population[ranking[POPULATION - 1]]);
                System.out.printf("%nReplaying best individual (fitness=%.3f) ...%n",
                        fitness[ranking[POPULATION - 1]]);
                client.setRenderMode(renderMode);   // arm renderer for this episode only
                runEpisode(client, net, nAgents, nEnvs, actionDim);
                client.setRenderMode("");            // disarm
                if ("rgb_array".equals(renderMode) && videoPath != null) {
                    client.saveVideo(videoPath);
                    System.out.println("Video saved to " + videoPath + ".mp4");
                }
            }
        }
    }

    // ── Episode rollout ───────────────────────────────────────────────────

    private static float runEpisode(SigmaRLClient client, Network net,
                                     int nAgents, int nEnvs, int actionDim) {
        StepResponse state = client.reset();
        float totalFitness = 0f;

        while (!SigmaRLClient.anyDone(state)) {
            // Build the flat actions array: [n_envs * n_agents * action_dim]
            float[] actions = new float[nEnvs * nAgents * actionDim];
            for (int env = 0; env < nEnvs; env++) {
                for (int a = 0; a < nAgents; a++) {
                    // Per-agent observation slice from StepResponse.observations
                    // (shape: [n_envs * n_agents * obs_dim], row-major).
                    // Typical content (default bridge config):
                    //   [self] forward speed, short-term ref path points,
                    //          distance to center line, distance to left/right boundary
                    //   [others] nearest-neighbor geometry (vertices), velocities,
                    //            distance-to-agent
                    // Optional fields depend on scenario flags (e.g., steering,
                    // other-agents' ref paths).
                    float[] obs    = SigmaRLClient.agentObs(state, env, a);
                    float[] action = net.forward(obs);
                    System.arraycopy(action, 0, actions,
                            (env * nAgents + a) * actionDim, actionDim);
                }
            }

            state = client.step(actions);

            // ── fitness function ──────────────────────────────────────────
            // Accumulate the simulator's own reward signal — it already encodes
            // collision avoidance, lane-following, and progress.
            // Students: replace with any scalar derived from agentReward().
            for (int env = 0; env < nEnvs; env++) {
                for (int a = 0; a < nAgents; a++) {
                    totalFitness += SigmaRLClient.agentReward(state, env, a);
                }
            }
        }
        return totalFitness;
    }

    // ── ES helpers ────────────────────────────────────────────────────────

    private static float[][] randomPopulation(int size, int nWeights, Random rng) {
        float[][] pop = new float[size][nWeights];
        for (float[] ind : pop) {
            for (int j = 0; j < nWeights; j++) {
                ind[j] = (float) rng.nextGaussian() * 0.1f;
            }
        }
        return pop;
    }

    private static float[] mutate(float[] weights, double std, Random rng) {
        float[] w = weights.clone();
        for (int i = 0; i < w.length; i++) {
            w[i] += (float) (rng.nextGaussian() * std);
        }
        return w;
    }

    private static int[] argsort(float[] arr) {
        Integer[] idx = new Integer[arr.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(arr[a], arr[b]));
        int[] out = new int[arr.length];
        for (int i = 0; i < out.length; i++) out[i] = idx[i];
        return out;
    }

    private static float mean(float[] arr) {
        float s = 0;
        for (float v : arr) s += v;
        return s / arr.length;
    }

    private static float[] toFloatArray(List<Float> list) {
        float[] arr = new float[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }

    private static void printObservationGuide(int obsDim) {
        System.out.println("Observation guide:");
        System.out.println("  obs = per-agent feature vector from Python scenario (not pixels)." );
        System.out.println("  Includes ego dynamics/path context + nearby-agent context.");
        System.out.printf ("  obs_dim reported by server: %d%n", obsDim);

        // Default bridge config is typically 32 for intersection_1:
        // ego_view=true, partial_observation=true, n_nearing_agents_observed=2,
        // n_points_short_term=3, observe_vertices=true,
        // observe_distance_to_agents=true, observe_distance_to_boundaries=true,
        // observe_distance_to_center_line=true.
        if (obsDim == 32) {
            System.out.println("  Default index map (obs_dim=32):");
            System.out.println("    [0]      ego forward speed");
            System.out.println("    [1..6]   ego short-term reference path (3 points x,y)");
            System.out.println("    [7]      ego distance to center line");
            System.out.println("    [8]      ego distance to left boundary");
            System.out.println("    [9]      ego distance to right boundary");
            System.out.println("    [10..20] nearest other agent features");
            System.out.println("    [21..31] second nearest other agent features");
            System.out.println("    per other-agent block (11):");
            System.out.println("      vertices(8) + velocity_xy(2) + distance_to_ego(1)");
        } else {
            System.out.println("  Layout is scenario/config dependent.");
            System.out.println("  Use SigmaRL obs provider flags to derive exact field order.");
        }
        System.out.println();
    }

    // ── Neural network  ────────────────────────────────────
    //
    // Single hidden layer:
    //   input  (obs_dim):    the full simulator observation per agent.
    //                        Relevant default fields include:
    //                        - ego forward speed (ego-frame x velocity)
    //                        - ego short-term reference path points
    //                        - ego distance to center line
    //                        - ego distance to left/right boundary
    //                        - nearest-agent geometry/velocity/distance
    //   hidden (HIDDEN_DIM): tanh activations
    //   output (action_dim): driving command, scaled to [actionLow, actionHigh]
    //
    // Swap this class for any other Java-based model — the EA loop is unchanged.

    static class Network {
        private final int     inputDim, hiddenDim, outputDim;
        private float[]       weights;  // flat: [W1 | b1 | W2 | b2]

        private final int offW1, offB1, offW2, offB2;

        // Per-output action bounds (from SpacesInfo)
        private final float[] outLow, outHigh;

        Network(int inputDim, int outputDim, int hiddenDim,
                float[] outLow, float[] outHigh) {
            this.inputDim  = inputDim;
            this.hiddenDim = hiddenDim;
            this.outputDim = outputDim;
            this.outLow    = outLow;
            this.outHigh   = outHigh;

            offW1 = 0;
            offB1 = offW1 + hiddenDim * inputDim;
            offW2 = offB1 + hiddenDim;
            offB2 = offW2 + outputDim * hiddenDim;

            this.weights = new float[weightCount()];
        }

        int weightCount() {
            return hiddenDim * inputDim    // W1
                 + hiddenDim              // b1
                 + outputDim * hiddenDim  // W2
                 + outputDim;             // b2
        }

        void loadWeights(float[] w) { this.weights = w; }

        float[] forward(float[] obs) {
            // Hidden layer
            float[] hidden = new float[hiddenDim];
            for (int h = 0; h < hiddenDim; h++) {
                float sum = weights[offB1 + h];
                for (int i = 0; i < inputDim; i++) {
                    sum += weights[offW1 + h * inputDim + i] * obs[i];
                }
                hidden[h] = (float) Math.tanh(sum);
            }

            // Output layer — scale tanh(-1,1) to per-action command range
            float[] out = new float[outputDim];
            for (int o = 0; o < outputDim; o++) {
                float sum = weights[offB2 + o];
                for (int h = 0; h < hiddenDim; h++) {
                    sum += weights[offW2 + o * hiddenDim + h] * hidden[h];
                }
                float t = (float) Math.tanh(sum);  // in (-1, 1)
                out[o] = outLow[o] + (t + 1f) * 0.5f * (outHigh[o] - outLow[o]);
            }
            return out;
        }
    }
}
