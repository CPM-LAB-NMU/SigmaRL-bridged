package io.sigmarl.bridge.example;

import io.sigmarl.bridge.PhysicalStepResponse;
import io.sigmarl.bridge.ScenarioConfig;
import io.sigmarl.bridge.SigmaRLClient;
import io.sigmarl.bridge.SpacesInfo;
import io.sigmarl.bridge.VehicleStateMsg;

import java.util.Arrays;
import java.util.Random;

/**
 * Genetic Algorithm demo using the <b>physical state interface</b> with a
 * fully <b>student-defined fitness function</b>.
 *
 * <p>This example deliberately uses a fitness metric that has nothing to do
 * with the simulator's internal reward signal: it measures
 * <em>survival time</em> — how many timesteps all agents remain active before
 * the episode ends.  Students are free to replace {@link #computeFitness} with
 * any function of the physical state (position, speed, heading, distance
 * travelled, etc.).
 *
 * <p>Contrast with {@link ExampleEvolutionaryRobotics}:
 * <ul>
 *   <li>{@link ExampleEvolutionaryRobotics} — (μ,λ)-ES, fitness = speed sum.</li>
 *   <li>This class — GA with elite carry-over + crossover, fitness = survival
 *       time.  Two different algorithm choices, two different fitness
 *       definitions, both running against the same simulator.</li>
 * </ul>
 *
 * <p>Run after starting the Python server:
 * <pre>
 *   python -m sigmarl.bridge.server
 *   java -cp target/sigmarl-java-client-*.jar io.sigmarl.bridge.example.ExampleEA
 * </pre>
 */
public class ExampleEA {

    // ── GA hyper-parameters ────────────────────────────────────────────────
    private static final int    POPULATION   = 20;
    private static final int    GENERATIONS  = 30;
    private static final int    ELITE_K      = 4;    // individuals carried unchanged
    private static final double MUTATION_STD = 0.05;
    private static final int    HIDDEN_DIM   = 32;

    // ── Physical dimensions (same vehicle as ExampleEvolutionaryRobotics) ─
    private static final int   STATE_DIM  = 4;   // [x, y, heading, speed]
    private static final int   ACTION_DIM = 2;   // [speed, curvature]
    private static final float MIN_SPEED  = -0.5f;
    private static final float MAX_SPEED  =  1.0f;
    private static final float MAX_CURV   =  4.0f;

    public static void main(String[] args) throws Exception {
        // Usage: [host] [port] [--render | --save-video PATH]
        String host       = "localhost";
        int    port       = 50051;
        String renderMode = "";
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

            ScenarioConfig cfg = SigmaRLClient.scenarioConfig(
                    "intersection_1", 4, /*n_envs=*/1, /*max_steps=*/128, "cpu");
            client.configure(cfg);

            SpacesInfo spaces = client.getSpaces();
            int nAgents = spaces.getNAgents();

            System.out.printf("Environment: %d agents%n", nAgents);
            System.out.printf("Fitness: survival time (steps completed)%n%n");

            Random rng  = new Random(42);
            Network net = new Network(STATE_DIM, ACTION_DIM, HIDDEN_DIM);

            int       nWeights  = net.weightCount();
            float[][] population = randomPopulation(POPULATION, nWeights, rng);
            float[]   fitness    = new float[POPULATION];

            for (int gen = 0; gen < GENERATIONS; gen++) {

                // ── evaluate ────────────────────────────────────────────────
                for (int i = 0; i < POPULATION; i++) {
                    net.loadWeights(population[i]);
                    fitness[i] = runEpisode(client, net, nAgents);
                }

                int[] ranking     = argsort(fitness);
                float bestFitness = fitness[ranking[POPULATION - 1]];
                float meanFitness = mean(fitness);
                System.out.printf("Gen %3d/%d | best=%.1f steps  mean=%.1f%n",
                        gen + 1, GENERATIONS, bestFitness, meanFitness);

                // ── build next generation ────────────────────────────────────
                float[][] next = new float[POPULATION][nWeights];

                // 1) carry elite individuals unchanged
                for (int k = 0; k < ELITE_K; k++) {
                    next[k] = population[ranking[POPULATION - 1 - k]].clone();
                }

                // 2) fill the rest with crossover + mutation from elite parents
                for (int i = ELITE_K; i < POPULATION; i++) {
                    int p1 = ranking[POPULATION - 1 - rng.nextInt(ELITE_K)];
                    int p2 = ranking[POPULATION - 1 - rng.nextInt(ELITE_K)];
                    next[i] = mutate(crossover(population[p1], population[p2], rng),
                                     MUTATION_STD, rng);
                }
                population = next;
            }

            System.out.println("\nEvolution complete.");

            // ── optional: replay best individual with rendering ───────────
            if (!renderMode.isEmpty()) {
                int[] ranking = argsort(fitness);
                net.loadWeights(population[ranking[POPULATION - 1]]);
                System.out.printf("%nReplaying best individual (fitness=%.1f steps) ...%n",
                        fitness[ranking[POPULATION - 1]]);
                client.setRenderMode(renderMode);
                runEpisode(client, net, nAgents);
                client.setRenderMode("");
                if ("rgb_array".equals(renderMode) && videoPath != null) {
                    client.saveVideo(videoPath);
                    System.out.println("Video saved to " + videoPath + ".mp4");
                }
            }
        }
    }

    // ── Episode rollout ───────────────────────────────────────────────────

    private static float runEpisode(SigmaRLClient client, Network net, int nAgents) {
        PhysicalStepResponse state = client.resetPhysical();
        float totalFitness = 0f;

        while (!SigmaRLClient.anyDonePhysical(state)) {
            int[]   ids        = new int[nAgents];
            float[] speeds     = new float[nAgents];
            float[] curvatures = new float[nAgents];

            for (int a = 0; a < nAgents; a++) {
                VehicleStateMsg s = SigmaRLClient.agentState(state, a);
                float[] input = {
                    (float) s.getX(),
                    (float) s.getY(),
                    (float) s.getHeading(),
                    (float) s.getSpeed()
                };
                float[] action = net.forward(input);
                ids[a]        = s.getVehicleId();
                speeds[a]     = action[0];
                curvatures[a] = action[1];
            }

            state = client.stepPhysical(ids, speeds, curvatures);

            // ── fitness function ──────────────────────────────────────────
            totalFitness += computeFitness(state, nAgents);
        }
        return totalFitness;
    }

    /**
     * Fitness contribution from one timestep: <b>survival time</b>.
     *
     * <p>Each timestep all agents are still active contributes 1.0 to fitness.
     * An episode that runs to the maximum step limit scores higher than one
     * that ends early (e.g. due to collision or leaving the map).
     *
     * <p>Students: replace this with any function of the physical state.
     * The {@code PhysicalStepResponse} gives {@code x, y, heading, speed}
     * for every agent at every timestep — use whatever makes sense for your
     * control objective.
     */
    private static float computeFitness(PhysicalStepResponse resp, int nAgents) {
        // +1 for every agent still in the episode this step
        float alive = 0f;
        for (int a = 0; a < nAgents; a++) {
            VehicleStateMsg s = SigmaRLClient.agentState(resp, a);
            if (s.getSpeed() >= 0f) alive += 1f;  // basic liveness check
        }
        return alive;
    }

    // ── GA helpers ────────────────────────────────────────────────────────

    private static float[][] randomPopulation(int size, int nWeights, Random rng) {
        float[][] pop = new float[size][nWeights];
        for (float[] ind : pop) {
            for (int j = 0; j < nWeights; j++) {
                ind[j] = (float) rng.nextGaussian() * 0.1f;
            }
        }
        return pop;
    }

    /** Uniform crossover: each gene taken independently from either parent. */
    private static float[] crossover(float[] p1, float[] p2, Random rng) {
        float[] child = new float[p1.length];
        for (int i = 0; i < p1.length; i++) {
            child[i] = rng.nextBoolean() ? p1[i] : p2[i];
        }
        return child;
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

    // ── Neural network (fully in Java) ────────────────────────────────────
    //
    // Identical architecture to ExampleEvolutionaryRobotics.Network.
    // Students can swap this for any Java model — the GA loop is unaffected.

    static class Network {
        private final int   inputDim, hiddenDim, outputDim;
        private float[]     weights;

        private final int offW1, offB1, offW2, offB2;

        private static final float[] OUT_LOW  = { MIN_SPEED, -MAX_CURV };
        private static final float[] OUT_HIGH = { MAX_SPEED,  MAX_CURV };

        Network(int inputDim, int outputDim, int hiddenDim) {
            this.inputDim  = inputDim;
            this.hiddenDim = hiddenDim;
            this.outputDim = outputDim;

            offW1 = 0;
            offB1 = offW1 + hiddenDim * inputDim;
            offW2 = offB1 + hiddenDim;
            offB2 = offW2 + outputDim * hiddenDim;

            this.weights = new float[weightCount()];
        }

        int weightCount() {
            return hiddenDim * inputDim
                 + hiddenDim
                 + outputDim * hiddenDim
                 + outputDim;
        }

        void loadWeights(float[] w) { this.weights = w; }

        float[] forward(float[] state) {
            float[] hidden = new float[hiddenDim];
            for (int h = 0; h < hiddenDim; h++) {
                float sum = weights[offB1 + h];
                for (int i = 0; i < inputDim; i++) {
                    sum += weights[offW1 + h * inputDim + i] * state[i];
                }
                hidden[h] = (float) Math.tanh(sum);
            }

            float[] out = new float[outputDim];
            for (int o = 0; o < outputDim; o++) {
                float sum = weights[offB2 + o];
                for (int h = 0; h < hiddenDim; h++) {
                    sum += weights[offW2 + o * hiddenDim + h] * hidden[h];
                }
                float t = (float) Math.tanh(sum);
                out[o] = OUT_LOW[o] + (t + 1f) * 0.5f * (OUT_HIGH[o] - OUT_LOW[o]);
            }
            return out;
        }
    }
}
