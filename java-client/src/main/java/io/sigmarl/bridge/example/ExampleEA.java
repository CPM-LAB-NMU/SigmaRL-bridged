package io.sigmarl.bridge.example;

import io.sigmarl.bridge.EvaluateResponse;
import io.sigmarl.bridge.SigmaRLClient;
import io.sigmarl.bridge.ScenarioConfig;

import java.util.Arrays;
import java.util.Random;

/**
 * Minimal evolutionary algorithm (simple GA) using SigmaRL as the fitness oracle.
 *
 * <p>Java owns the entire learning loop:
 * <ol>
 *   <li>Initialise a population of random weight vectors.</li>
 *   <li>Evaluate each individual by calling {@code EvaluateWeights} — Python runs
 *       the full episodes and returns cumulative reward.</li>
 *   <li>Select the top-k individuals, apply Gaussian mutation, repeat.</li>
 * </ol>
 *
 * <p>Run after starting the Python server:
 * <pre>
 *   python -m sigmarl.bridge.server
 *   java -cp target/sigmarl-java-client-*.jar io.sigmarl.bridge.example.ExampleEA
 * </pre>
 */
public class ExampleEA {

    // ── GA hyper-parameters ────────────────────────────────────────────────
    private static final int    POPULATION    = 20;
    private static final int    GENERATIONS   = 30;
    private static final int    EPISODES      = 3;    // per individual per generation
    private static final int    ELITE_K       = 4;    // survivors per generation
    private static final double MUTATION_STD  = 0.05;

    public static void main(String[] args) throws Exception {
        String host = args.length > 0 ? args[0] : "localhost";
        int    port = args.length > 1 ? Integer.parseInt(args[1]) : 50051;

        try (SigmaRLClient client = new SigmaRLClient(host, port)) {

            // Optional: override default scenario
            ScenarioConfig cfg = SigmaRLClient.scenarioConfig(
                    "intersection_1", 4, /*n_envs=*/4, /*max_steps=*/128, "cpu");
            client.configure(cfg);

            long nWeights = client.getWeightCount();
            System.out.printf("Policy parameter count: %d%n", nWeights);

            Random rng = new Random(42);
            float[][] population = initPopulation(POPULATION, (int) nWeights, rng);
            float[]   fitness    = new float[POPULATION];

            for (int gen = 0; gen < GENERATIONS; gen++) {
                // ── evaluate ────────────────────────────────────────────────
                for (int i = 0; i < POPULATION; i++) {
                    EvaluateResponse resp = client.evaluateWeights(population[i], EPISODES);
                    fitness[i] = resp.getMeanReward();
                }

                int[] ranking = argsort(fitness);
                float bestFitness = fitness[ranking[POPULATION - 1]];
                float meanFitness = mean(fitness);
                System.out.printf("Gen %3d | best=%.3f  mean=%.3f%n",
                        gen, bestFitness, meanFitness);

                // ── select elite + mutate ───────────────────────────────────
                float[][] nextPop = new float[POPULATION][(int) nWeights];
                for (int k = 0; k < ELITE_K; k++) {
                    // carry elite individuals unchanged
                    nextPop[k] = population[ranking[POPULATION - 1 - k]].clone();
                }
                for (int i = ELITE_K; i < POPULATION; i++) {
                    // pick a random elite parent and mutate
                    int parentIdx = ranking[POPULATION - 1 - (rng.nextInt(ELITE_K))];
                    nextPop[i] = mutate(population[parentIdx], MUTATION_STD, rng);
                }
                population = nextPop;
            }

            System.out.println("Evolution complete.");
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private static float[][] initPopulation(int size, int nWeights, Random rng) {
        float[][] pop = new float[size][nWeights];
        for (float[] individual : pop) {
            for (int j = 0; j < nWeights; j++) {
                individual[j] = (float) rng.nextGaussian() * 0.1f;
            }
        }
        return pop;
    }

    private static float[] mutate(float[] weights, double std, Random rng) {
        float[] mutated = weights.clone();
        for (int i = 0; i < mutated.length; i++) {
            mutated[i] += (float) (rng.nextGaussian() * std);
        }
        return mutated;
    }

    /** Returns indices that would sort {@code arr} in ascending order. */
    private static int[] argsort(float[] arr) {
        Integer[] idx = new Integer[arr.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(arr[a], arr[b]));
        int[] result = new int[arr.length];
        for (int i = 0; i < result.length; i++) result[i] = idx[i];
        return result;
    }

    private static float mean(float[] arr) {
        float sum = 0;
        for (float v : arr) sum += v;
        return sum / arr.length;
    }
}
