package io.sigmarl.bridge.example;

import io.sigmarl.bridge.ScenarioConfig;
import io.sigmarl.bridge.SigmaRLClient;
import io.sigmarl.bridge.SpacesInfo;
import io.sigmarl.bridge.StepResponse;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Genetic Algorithm demo with a <b>student-defined fitness function</b>.
 *
 * <p>This example deliberately uses a fitness metric that has nothing to do
 * with the simulator's internal reward signal: it measures
 * <em>survival time</em> — how many timesteps the episode runs before any
 * environment signals done.  A controller that avoids all collisions and
 * stays on the map for all 128 steps scores the maximum.
 *
 * <p>Students are free to replace {@link #computeFitness} with any scalar
 * derived from the {@code StepResponse} (observations, rewards, step count,
 * etc.).
 *
 * <p>Contrast with {@link ExampleEvolutionaryRobotics}:
 * <ul>
 *   <li>{@link ExampleEvolutionaryRobotics} — (μ,λ)-ES, fitness = cumulative reward.</li>
 *   <li>This class — GA with elite carry-over + crossover, fitness = survival time.</li>
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
    private static final int    HIDDEN_DIM   = 64;

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

            SpacesInfo spaces    = client.getSpaces();
            int   nAgents   = spaces.getNAgents();
            int   nEnvs     = spaces.getNEnvs();
            int   obsDim    = spaces.getObsDim();
            int   actionDim = spaces.getActionDim();
            float[] actionLow  = toFloatArray(spaces.getActionLowList());
            float[] actionHigh = toFloatArray(spaces.getActionHighList());

            System.out.printf("Environment: %d agents | obs_dim=%d%n", nAgents, obsDim);
            System.out.printf("Fitness: survival time (steps completed)%n%n");

            Random rng  = new Random(42);
            Network net = new Network(obsDim, actionDim, HIDDEN_DIM, actionLow, actionHigh);

            int       nWeights   = net.weightCount();
            float[][] population = randomPopulation(POPULATION, nWeights, rng);
            float[]   fitness    = new float[POPULATION];

            for (int gen = 0; gen < GENERATIONS; gen++) {

                // ── evaluate ────────────────────────────────────────────────
                for (int i = 0; i < POPULATION; i++) {
                    net.loadWeights(population[i]);
                    fitness[i] = runEpisode(client, net, nAgents, nEnvs, actionDim);
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
                runEpisode(client, net, nAgents, nEnvs, actionDim);
                client.setRenderMode("");
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
            float[] actions = new float[nEnvs * nAgents * actionDim];
            for (int env = 0; env < nEnvs; env++) {
                for (int a = 0; a < nAgents; a++) {
                    float[] obs    = SigmaRLClient.agentObs(state, env, a);
                    float[] action = net.forward(obs);
                    System.arraycopy(action, 0, actions,
                            (env * nAgents + a) * actionDim, actionDim);
                }
            }

            state = client.step(actions);

            // ── fitness function ──────────────────────────────────────────
            totalFitness += computeFitness(state);
        }
        return totalFitness;
    }

    /**
     * Fitness contribution from one timestep: <b>survival time</b>.
     *
     * <p>Returns 1.0 each timestep the episode is still running.
     * An episode that runs to the maximum step limit scores higher than one
     * that ends early (e.g. due to collision or leaving the map).
     *
     * <p>Students: replace this with any scalar derived from the
     * {@code StepResponse}.  Use {@link SigmaRLClient#agentObs} to read
     * individual observation vectors, or {@link SigmaRLClient#agentReward}
     * to use the simulator's built-in reward instead of step count.
     */
    private static float computeFitness(StepResponse resp) {
        return 1f;  // +1 for every step the episode is still alive
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

    private static float[] toFloatArray(List<Float> list) {
        float[] arr = new float[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }

    // ── Neural network (fully in Java) ────────────────────────────────────
    //
    // Identical architecture to ExampleEvolutionaryRobotics.Network.
    // Students can swap this for any Java model — the GA loop is unaffected.

    static class Network {
        private final int   inputDim, hiddenDim, outputDim;
        private float[]     weights;

        private final int offW1, offB1, offW2, offB2;
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
            return hiddenDim * inputDim
                 + hiddenDim
                 + outputDim * hiddenDim
                 + outputDim;
        }

        void loadWeights(float[] w) { this.weights = w; }

        float[] forward(float[] obs) {
            float[] hidden = new float[hiddenDim];
            for (int h = 0; h < hiddenDim; h++) {
                float sum = weights[offB1 + h];
                for (int i = 0; i < inputDim; i++) {
                    sum += weights[offW1 + h * inputDim + i] * obs[i];
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
                out[o] = outLow[o] + (t + 1f) * 0.5f * (outHigh[o] - outLow[o]);
            }
            return out;
        }
    }
}
