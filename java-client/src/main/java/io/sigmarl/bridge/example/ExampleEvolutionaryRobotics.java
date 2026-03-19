package io.sigmarl.bridge.example;

import io.sigmarl.bridge.ScenarioConfig;
import io.sigmarl.bridge.SigmaRLClient;
import io.sigmarl.bridge.SpacesInfo;
import io.sigmarl.bridge.StepResponse;

import java.util.Arrays;
import java.util.Random;

/**
 * Evolutionary Robotics demo: Java defines and runs the neural network;
 * Python provides the multi-agent driving environment.
 *
 * <p>This is the key difference from {@link ExampleEA}:
 * <ul>
 *   <li>{@link ExampleEA} sends weight vectors to Python and lets Python run
 *       the forward pass inside {@code EvaluateWeights}.</li>
 *   <li>This demo uses {@code Reset}/{@code Step} so the <b>Java network does
 *       every forward pass</b>. The architecture is fully under Java control
 *       and can be swapped for DL4J, ONNX Runtime, etc.</li>
 * </ul>
 *
 * <p>Algorithm: simple (μ, λ)-ES with Gaussian mutation.
 * <pre>
 *   for each generation:
 *     for each individual in population:
 *       load weights into Java network
 *       run one episode via Reset / Step
 *       record total reward as fitness
 *     select top-μ survivors
 *     generate λ offspring by mutating survivors
 * </pre>
 *
 * <p>Run with the Python server active:
 * <pre>
 *   python -m sigmarl.bridge.server
 *   java -cp target/sigmarl-java-client-*.jar \
 *        io.sigmarl.bridge.example.ExampleEvolutionaryRobotics
 * </pre>
 */
public class ExampleEvolutionaryRobotics {

    // ── ES hyper-parameters ───────────────────────────────────────────────
    private static final int    POPULATION   = 20;   // λ — offspring per generation
    private static final int    SURVIVORS    = 5;    // μ — parents selected each gen
    private static final int    GENERATIONS  = 50;
    private static final double MUTATION_STD = 0.05; // Gaussian noise σ
    private static final int    HIDDEN_DIM   = 64;   // Java network hidden layer size
    private static final int    LOG_EVERY    = 5;

    public static void main(String[] args) throws Exception {
        String host = args.length > 0 ? args[0] : "localhost";
        int    port = args.length > 1 ? Integer.parseInt(args[1]) : 50051;

        try (SigmaRLClient client = new SigmaRLClient(host, port)) {

            // Configure scenario
            ScenarioConfig cfg = SigmaRLClient.scenarioConfig(
                    "intersection_1", 4, /*n_envs=*/1, /*max_steps=*/128, "cpu");
            client.configure(cfg);

            SpacesInfo spaces = client.getSpaces();
            int nAgents   = spaces.getNAgents();
            int obsDim    = spaces.getObsDim();
            int actionDim = spaces.getActionDim();

            System.out.printf(
                    "Environment: %d agents | obs_dim=%d | action_dim=%d%n",
                    nAgents, obsDim, actionDim);
            System.out.printf(
                    "Network: %d -> %d -> %d (per agent, shared weights)%n%n",
                    obsDim, HIDDEN_DIM, actionDim);

            // Read the actual physical action bounds from the environment
            float[] actionLow  = toFloatArray(spaces.getActionLowList());
            float[] actionHigh = toFloatArray(spaces.getActionHighList());

            Random rng = new Random(42);
            Network net = new Network(obsDim, actionDim, HIDDEN_DIM, actionLow, actionHigh);

            // Initialise population as flat weight vectors
            int nWeights = net.weightCount();
            float[][] population = randomPopulation(POPULATION, nWeights, rng);
            float[]   fitness    = new float[POPULATION];

            for (int gen = 0; gen < GENERATIONS; gen++) {

                // ── evaluate each individual ─────────────────────────────
                for (int i = 0; i < POPULATION; i++) {
                    net.loadWeights(population[i]);
                    fitness[i] = runEpisode(client, net, nAgents, actionDim);
                }

                // ── report ───────────────────────────────────────────────
                int[] ranking = argsort(fitness);  // ascending
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
        }
    }

    // ── Episode rollout (Java runs the forward pass) ──────────────────────

    private static float runEpisode(
            SigmaRLClient client, Network net, int nAgents, int actionDim) {

        StepResponse state = client.reset();
        float totalReward  = 0f;

        while (!SigmaRLClient.anyDone(state)) {
            float[] actions = new float[nAgents * actionDim];
            for (int a = 0; a < nAgents; a++) {
                float[] obs    = SigmaRLClient.agentObs(state, /*env=*/0, a);
                float[] action = net.forward(obs);
                System.arraycopy(action, 0, actions, a * actionDim, actionDim);
            }

            StepResponse next = client.step(actions);

            for (int a = 0; a < nAgents; a++) {
                totalReward += SigmaRLClient.agentReward(next, 0, a);
            }
            state = next;
        }
        return totalReward;
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

    /** Returns indices that would sort {@code arr} ascending. */
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
    // Single hidden layer: obs_dim -> tanh -> hidden_dim -> tanh -> action_dim
    //
    // Swap this class for a DL4J MultiLayerNetwork, an ONNX Runtime session,
    // or any other Java-based model — the EA loop above is unchanged.

    static class Network {
        private final int     inputDim, hiddenDim, outputDim;
        private final float[] actionLow, actionHigh;  // physical bounds from env
        private float[]       weights;                // flat: [W1 | b1 | W2 | b2]

        // offsets into the flat weight vector
        private final int offW1, offB1, offW2, offB2;

        Network(int inputDim, int outputDim, int hiddenDim,
                float[] actionLow, float[] actionHigh) {
            this.inputDim   = inputDim;
            this.hiddenDim  = hiddenDim;
            this.outputDim  = outputDim;
            this.actionLow  = actionLow;
            this.actionHigh = actionHigh;

            offW1 = 0;
            offB1 = offW1 + hiddenDim * inputDim;
            offW2 = offB1 + hiddenDim;
            offB2 = offW2 + outputDim * hiddenDim;

            this.weights = new float[weightCount()];
        }

        int weightCount() {
            return hiddenDim * inputDim   // W1
                 + hiddenDim             // b1
                 + outputDim * hiddenDim // W2
                 + outputDim;            // b2
        }

        void loadWeights(float[] w) {
            this.weights = w;
        }

        float[] getWeights() {
            return weights.clone();
        }

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

            // Output layer — tanh in (-1, 1) then scaled to physical action bounds
            float[] out = new float[outputDim];
            for (int o = 0; o < outputDim; o++) {
                float sum = weights[offB2 + o];
                for (int h = 0; h < hiddenDim; h++) {
                    sum += weights[offW2 + o * hiddenDim + h] * hidden[h];
                }
                float t = (float) Math.tanh(sum);  // in (-1, 1)
                // scale to [actionLow[o], actionHigh[o]]
                out[o] = actionLow[o] + (t + 1f) * 0.5f * (actionHigh[o] - actionLow[o]);
            }
            return out;
        }
    }

    private static float[] toFloatArray(java.util.List<Float> list) {
        float[] arr = new float[list.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = list.get(i);
        return arr;
    }
}
