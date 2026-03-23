package io.sigmarl.bridge.example;

import io.sigmarl.bridge.PhysicalStepResponse;
import io.sigmarl.bridge.ScenarioConfig;
import io.sigmarl.bridge.SigmaRLClient;
import io.sigmarl.bridge.SpacesInfo;
import io.sigmarl.bridge.VehicleStateMsg;

import java.util.Arrays;
import java.util.Random;

/**
 * Evolutionary Robotics demo using the <b>physical state interface</b>.
 *
 * <p>Students implement every component from scratch:
 * <ol>
 *   <li><b>Neural network</b> — takes interpretable physical state per agent
 *       ({@code x, y, heading, speed}) and outputs a driving command
 *       ({@code speed, curvature}).  No opaque observation tensors.</li>
 *   <li><b>Evolutionary algorithm</b> — (μ, λ)-ES with Gaussian mutation.</li>
 *   <li><b>Fitness function</b> — total reward accumulated across all agents.
 *       Change {@link #computeFitness} to use any combination of the physical
 *       state signals returned by {@code StepPhysical}.</li>
 * </ol>
 *
 * <p>The Python server acts purely as a physics simulator.  It returns
 * per-vehicle lab-frame state ({@code x, y, heading, speed}) and advances the
 * world when given speed + curvature commands — exactly the same interface a
 * student's rosbridge would use against real hardware.
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
    private static final int    HIDDEN_DIM   = 64;
    private static final int    LOG_EVERY    = 5;

    // ── Physical state / command dimensions ───────────────────────────────
    // Each agent's input: [x, y, heading, speed]
    private static final int STATE_DIM  = 4;
    // Each agent's output: [speed, curvature]
    private static final int ACTION_DIM = 2;

    // Physical command bounds for this vehicle (CPM lab miniature cars).
    // speed    ∈ [MIN_SPEED,  MAX_SPEED]  m/s
    // curvature∈ [-MAX_CURV,  MAX_CURV]   1/m  (positive = left turn)
    //   derived from max steering angle (31°) and wheelbase (0.15 m):
    //   max_curv = tan(31° * π/180) / 0.15 ≈ 4.0 m⁻¹
    private static final float MIN_SPEED = -0.5f;
    private static final float MAX_SPEED =  1.0f;
    private static final float MAX_CURV  =  4.0f;

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
            // Rendering is armed only for the final replay via setRenderMode().
            ScenarioConfig cfg = SigmaRLClient.scenarioConfig(
                    "intersection_1", 4, /*n_envs=*/1, /*max_steps=*/128, "cpu");
            client.configure(cfg);

            SpacesInfo spaces = client.getSpaces();
            int nAgents = spaces.getNAgents();

            System.out.printf(
                    "Environment: %d agents | state_dim=%d | action_dim=%d%n",
                    nAgents, STATE_DIM, ACTION_DIM);
            System.out.printf(
                    "Network per agent: %d -> %d -> %d%n%n",
                    STATE_DIM, HIDDEN_DIM, ACTION_DIM);

            Random rng  = new Random(42);
            Network net = new Network(STATE_DIM, ACTION_DIM, HIDDEN_DIM);

            int     nWeights  = net.weightCount();
            float[][] population = randomPopulation(POPULATION, nWeights, rng);
            float[]   fitness    = new float[POPULATION];

            for (int gen = 0; gen < GENERATIONS; gen++) {

                // ── evaluate each individual ─────────────────────────────
                for (int i = 0; i < POPULATION; i++) {
                    net.loadWeights(population[i]);
                    fitness[i] = runEpisode(client, net, nAgents);
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
                runEpisode(client, net, nAgents);
                client.setRenderMode("");            // disarm
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

                // Build the physical input vector — fully interpretable
                float[] input = {
                    (float) s.getX(),
                    (float) s.getY(),
                    (float) s.getHeading(),
                    (float) s.getSpeed()
                };

                float[] action = net.forward(input);  // [speed, curvature]

                ids[a]        = s.getVehicleId();
                speeds[a]     = action[0];
                curvatures[a] = action[1];
            }

            PhysicalStepResponse next = client.stepPhysical(ids, speeds, curvatures);

            // ── fitness function (students define this) ───────────────────
            // Currently: sum the speed of all agents as a proxy for progress.
            // Replace with any combination of x, y, heading, speed you want.
            totalFitness += computeFitness(next, nAgents);

            state = next;
        }
        return totalFitness;
    }

    /**
     * Fitness contribution from one timestep.
     *
     * <p>This is the method students are meant to customise.  The
     * {@code PhysicalStepResponse} gives the full lab-frame state of every
     * agent — position, heading, speed — at each timestep.
     *
     * <p>Example alternatives:
     * <ul>
     *   <li>Maximise average speed: {@code state.getSpeed()}</li>
     *   <li>Penalise deviations from a target position</li>
     *   <li>Count steps without leaving a bounding box</li>
     *   <li>Minimise energy: {@code -Math.abs(speed)}</li>
     * </ul>
     */
    private static float computeFitness(PhysicalStepResponse resp, int nAgents) {
        float contribution = 0f;
        for (int a = 0; a < nAgents; a++) {
            VehicleStateMsg s = SigmaRLClient.agentState(resp, a);
            contribution += s.getSpeed();  // reward forward progress
        }
        return contribution;
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

    // ── Neural network (fully in Java) ────────────────────────────────────
    //
    // Single hidden layer:
    //   input  (STATE_DIM=4): [x, y, heading, speed]  — physical lab-frame state
    //   hidden (HIDDEN_DIM):  tanh activations
    //   output (ACTION_DIM=2): [speed, curvature]      — physical driving command
    //
    // Output is scaled from tanh range (-1, 1) to the physical command bounds
    // defined by MIN_SPEED/MAX_SPEED and ±MAX_CURV.
    //
    // Swap this class for any other Java-based model — the EA loop is unchanged.

    static class Network {
        private final int     inputDim, hiddenDim, outputDim;
        private float[]       weights;  // flat: [W1 | b1 | W2 | b2]

        private final int offW1, offB1, offW2, offB2;

        // Physical command bounds for the two output neurons
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
            return hiddenDim * inputDim    // W1
                 + hiddenDim              // b1
                 + outputDim * hiddenDim  // W2
                 + outputDim;             // b2
        }

        void loadWeights(float[] w) { this.weights = w; }

        float[] forward(float[] state) {
            // Hidden layer
            float[] hidden = new float[hiddenDim];
            for (int h = 0; h < hiddenDim; h++) {
                float sum = weights[offB1 + h];
                for (int i = 0; i < inputDim; i++) {
                    sum += weights[offW1 + h * inputDim + i] * state[i];
                }
                hidden[h] = (float) Math.tanh(sum);
            }

            // Output layer — scale tanh(-1,1) to physical command range
            float[] out = new float[outputDim];
            for (int o = 0; o < outputDim; o++) {
                float sum = weights[offB2 + o];
                for (int h = 0; h < hiddenDim; h++) {
                    sum += weights[offW2 + o * hiddenDim + h] * hidden[h];
                }
                float t = (float) Math.tanh(sum);  // in (-1, 1)
                out[o] = OUT_LOW[o] + (t + 1f) * 0.5f * (OUT_HIGH[o] - OUT_LOW[o]);
            }
            return out;
        }
    }
}
