package io.sigmarl.bridge.example;

import io.sigmarl.bridge.ScenarioConfig;
import io.sigmarl.bridge.SigmaRLClient;
import io.sigmarl.bridge.SpacesInfo;
import io.sigmarl.bridge.StepResponse;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Evolutionary Robotics demo — <b>custom fitness function, no RL reward</b>.
 *
 * <p>This example uses the simulator purely as a physics engine.  The fitness
 * function is computed entirely from the per-agent observation vector returned
 * by {@link SigmaRLClient#agentObs} — it never calls
 * {@link SigmaRLClient#agentReward}, which would hand the RL reward back from
 * the Python side.
 *
 * <p>The fitness mirrors what the RL reward encodes (forward progress,
 * path following, collision avoidance) but it is written from scratch in Java,
 * reading named fields out of the observation vector.  Students can see exactly
 * what each term does and swap or extend it freely.
 *
 * <h3>Fitness terms and their RL reward counterparts</h3>
 * <table border="1">
 *   <tr><th>Fitness term</th><th>Obs index</th><th>RL reward analogue</th></tr>
 *   <tr><td>Forward speed reward</td><td>obs[0]</td>
 *       <td>{@code reward_movement} + {@code reward_vel}</td></tr>
 *   <tr><td>Path deviation penalty</td><td>obs[7]</td>
 *       <td>{@code penalty_deviate_from_ref_path}</td></tr>
 *   <tr><td>Boundary proximity penalty</td><td>obs[8], obs[9]</td>
 *       <td>{@code penalty_close_to_lanelets}</td></tr>
 *   <tr><td>Agent proximity penalty</td><td>obs[20], obs[31]</td>
 *       <td>{@code penalty_close_to_agents}</td></tr>
 *   <tr><td>Collision proxy penalty</td><td>obs[20] &lt; threshold</td>
 *       <td>{@code penalty_collide_other_agents} + {@code penalty_collide_with_boundaries}</td></tr>
 * </table>
 *
 * <p>Observation index layout (default config: obs_dim=32, is_obs_steering=false):
 * <pre>
 *   [0]      ego forward speed (normalised by max_speed; ~0..1)
 *   [1..6]   ego short-term reference path — 3 look-ahead points (x, y each)
 *   [7]      ego distance to center line (normalised; 0 = on path)
 *   [8]      ego distance to left boundary  (normalised; 0 = at wall)
 *   [9]      ego distance to right boundary (normalised; 0 = at wall)
 *   [10..19] nearest other-agent block: 8 corner vertices (x,y) + velocity (x,y)
 *   [20]     distance to nearest other agent (normalised; 0 = touching)
 *   [21..30] second nearest other-agent block: same layout as above
 *   [31]     distance to second nearest other agent (normalised)
 * </pre>
 *
 * Distances are normalised by {@code lane_width × 3}, so a value of 0.33
 * corresponds to approximately one lane width.
 *
 * <p>Algorithm: (μ, λ)-Evolution Strategy — identical to
 * {@link ExampleEvolutionaryRobotics}.  Only the fitness function differs.
 *
 * <p>Run after starting the Python gRPC server:
 * <pre>
 *   python -m sigmarl.bridge.server
 *   java -cp target/sigmarl-java-client-*.jar \
 *        io.sigmarl.bridge.example.ExampleEvolutionaryRoboticsCustomFitness
 *   # with live rendering of the best individual at the end:
 *   java -cp target/sigmarl-java-client-*.jar \
 *        io.sigmarl.bridge.example.ExampleEvolutionaryRoboticsCustomFitness --render
 * </pre>
 */
public class ExampleEvolutionaryRoboticsCustomFitness {

    // ── ES hyper-parameters ───────────────────────────────────────────────
    private static final int    POPULATION   = 20;
    private static final int    SURVIVORS    = 5;
    private static final int    GENERATIONS  = 50;
    private static final double MUTATION_STD = 0.05;
    private static final int    HIDDEN_DIM   = 64;
    private static final int    LOG_EVERY    = 5;

    // ── Fitness function weights ──────────────────────────────────────────
    //
    // These scale each term of computeFitness() to give a balanced signal.
    // All obs values are normalised to roughly [0, 1], so weights here act
    // as relative importance scores.  Students: tune these freely.

    /** Reward for moving forward fast.  Mirrors reward_movement + reward_vel. */
    private static final float W_SPEED       = 1.0f;

    /** Penalty for drifting off the reference path.  Mirrors penalty_deviate_from_ref_path. */
    private static final float W_DEVIATION   = 1.0f;

    /** Penalty for being close to lane boundaries.  Mirrors penalty_close_to_lanelets. */
    private static final float W_BOUNDARY    = 1.5f;

    /** Penalty for being close to other agents.  Mirrors penalty_close_to_agents. */
    private static final float W_PROXIMITY   = 1.5f;

    /**
     * Hard penalty applied when an agent appears to collide (obs distance < threshold).
     * Mirrors penalty_collide_other_agents + penalty_collide_with_boundaries.
     */
    private static final float W_COLLISION   = 5.0f;

    /**
     * Normalised distance below which we treat two agents as colliding.
     * (lane_width × 3 normaliser, so 0.05 ≈ 5% of three lane widths ≈ touching bumpers.)
     */
    private static final float COLLISION_DIST = 0.05f;

    /**
     * Normalised distance below which boundary proximity penalty activates.
     * 0.33 ≈ one lane width — gives comfortable margin before the wall.
     */
    private static final float BOUNDARY_SAFE = 0.33f;

    /**
     * Normalised distance below which agent proximity penalty activates.
     * Same scale as BOUNDARY_SAFE.
     */
    private static final float AGENT_SAFE    = 0.33f;

    // ── Obs-vector index constants ────────────────────────────────────────

    private static final int IDX_SPEED        =  0;
    private static final int IDX_DIST_CENTER  =  7;
    private static final int IDX_DIST_LEFT    =  8;
    private static final int IDX_DIST_RIGHT   =  9;
    private static final int IDX_DIST_AGENT0  = 20;  // end of 1st other-agent block
    private static final int IDX_DIST_AGENT1  = 31;  // end of 2nd other-agent block

    // ── Entry point ───────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        // Usage: [host] [port] [--render | --save-video PATH]
        String host       = "localhost";
        int    port       = 50051;
        String renderMode = "";
        String videoPath  = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--render":     renderMode = "human";    break;
                case "--save-video": renderMode = "rgb_array";
                                    videoPath  = args[++i];  break;
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

            System.out.printf(
                    "Environment: %d agents | obs_dim=%d | action_dim=%d%n",
                    nAgents, obsDim, actionDim);
            System.out.println("Fitness: custom obs-based (no agentReward() call)");
            System.out.println();
            printFitnessGuide(obsDim);

            Random  rng  = new Random(42);
            Network net  = new Network(obsDim, actionDim, HIDDEN_DIM, actionLow, actionHigh);

            int       nWeights   = net.weightCount();
            float[][] population = randomPopulation(POPULATION, nWeights, rng);
            float[]   fitness    = new float[POPULATION];

            for (int gen = 0; gen < GENERATIONS; gen++) {

                // ── evaluate each individual ─────────────────────────────
                for (int i = 0; i < POPULATION; i++) {
                    net.loadWeights(population[i]);
                    fitness[i] = runEpisode(client, net, nAgents, nEnvs, obsDim, actionDim);
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
                client.setRenderMode(renderMode);
                runEpisode(client, net, nAgents, nEnvs, obsDim, actionDim);
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
                                     int nAgents, int nEnvs, int obsDim, int actionDim) {
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
            // Accumulate per-agent fitness from the observation vector only.
            // agentReward() is intentionally NOT called here.
            for (int env = 0; env < nEnvs; env++) {
                for (int a = 0; a < nAgents; a++) {
                    float[] obs = SigmaRLClient.agentObs(state, env, a);
                    totalFitness += computeFitness(obs, obsDim);
                }
            }
        }
        return totalFitness;
    }

    // ── Fitness function ──────────────────────────────────────────────────

    /**
     * Compute the fitness contribution for one agent at one timestep,
     * reading only from the observation vector.
     *
     * <p>Each term below is the Java equivalent of a component in
     * {@code ScenarioRoadTraffic.reward()} in the Python scenario, re-derived
     * from the normalised values already present in the observation.
     *
     * <p>All distances in the obs vector are normalised by
     * {@code lane_width × 3}, so 1.0 means "three lane widths away" and
     * 0.0 means "at the boundary / touching another vehicle".
     *
     * @param obs    per-agent observation slice from {@link SigmaRLClient#agentObs}
     * @param obsDim full observation dimension (used for bounds checking)
     * @return scalar fitness contribution for this timestep
     */
    static float computeFitness(float[] obs, int obsDim) {
        float f = 0f;

        // ── Term 1: forward speed reward ─────────────────────────────────
        // obs[0] = ego forward speed normalised by max_speed.
        // Higher speed in the forward direction scores more fitness.
        // Directly analogous to reward_movement + reward_vel in the RL reward,
        // both of which reward progress along the reference path.
        float speed = safeGet(obs, IDX_SPEED, obsDim, 0f);
        f += W_SPEED * speed;

        // ── Term 2: path deviation penalty ───────────────────────────────
        // obs[7] = normalised distance from the vehicle centre to the
        // reference path centre line.  0 = perfectly on path, 1 = three
        // lane widths off.  Penalise proportionally to how far off-path
        // the vehicle drifts.
        // Analogous to penalty_deviate_from_ref_path in the RL reward.
        float distCenter = safeGet(obs, IDX_DIST_CENTER, obsDim, 0f);
        f -= W_DEVIATION * distCenter;

        // ── Term 3: boundary proximity penalty ───────────────────────────
        // obs[8] = normalised distance to left boundary.
        // obs[9] = normalised distance to right boundary.
        // softPenalty() returns 0 when the vehicle is safely away from the
        // wall (dist >= BOUNDARY_SAFE) and rises linearly as it approaches.
        // Analogous to penalty_close_to_lanelets (exponential in the RL
        // reward; linear here for simplicity — students can swap for exp).
        float distLeft  = safeGet(obs, IDX_DIST_LEFT,  obsDim, 1f);
        float distRight = safeGet(obs, IDX_DIST_RIGHT, obsDim, 1f);
        f -= W_BOUNDARY * softPenalty(distLeft,  BOUNDARY_SAFE);
        f -= W_BOUNDARY * softPenalty(distRight, BOUNDARY_SAFE);

        // Hard collision proxy: if the vehicle is extremely close to a wall,
        // treat it as a boundary collision and apply a large one-off penalty.
        // Analogous to penalty_collide_with_boundaries.
        if (distLeft < COLLISION_DIST || distRight < COLLISION_DIST) {
            f -= W_COLLISION;
        }

        // ── Term 4: agent proximity penalty + collision proxy ─────────────
        // obs[20] = normalised distance to the nearest other agent.
        // obs[31] = normalised distance to the second nearest other agent.
        // Same soft-threshold logic as the boundary term above.
        // Analogous to penalty_close_to_agents + penalty_collide_other_agents.
        float distAgent0 = safeGet(obs, IDX_DIST_AGENT0, obsDim, 1f);
        float distAgent1 = safeGet(obs, IDX_DIST_AGENT1, obsDim, 1f);
        f -= W_PROXIMITY * softPenalty(distAgent0, AGENT_SAFE);
        f -= W_PROXIMITY * softPenalty(distAgent1, AGENT_SAFE);

        // Hard collision proxy for inter-agent collision.
        if (distAgent0 < COLLISION_DIST) {
            f -= W_COLLISION;
        }

        return f;
    }

    /**
     * Soft proximity penalty: 0 when {@code dist >= safe}, rising linearly
     * to 1 as {@code dist} approaches 0.
     *
     * <p>This is a linear approximation of the exponential-decreasing function
     * used in the RL reward.  Replace with {@code (float) Math.exp(-dist / safe)}
     * for a smoother gradient if desired.
     */
    private static float softPenalty(float dist, float safe) {
        if (dist >= safe) return 0f;
        return (safe - dist) / safe;
    }

    /** Safe array access with a default value when the index is out of bounds. */
    private static float safeGet(float[] arr, int idx, int dim, float def) {
        return (idx < dim && idx < arr.length) ? arr[idx] : def;
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

    // ── Diagnostics ───────────────────────────────────────────────────────

    private static void printFitnessGuide(int obsDim) {
        System.out.println("Custom fitness function — obs indices used:");
        System.out.printf("  [%d]      ego forward speed          -> speed reward (W=%.1f)%n",
                IDX_SPEED, W_SPEED);
        System.out.printf("  [%d]      distance to center line   -> deviation penalty (W=%.1f)%n",
                IDX_DIST_CENTER, W_DEVIATION);
        System.out.printf("  [%d]      distance to left boundary -> boundary penalty (W=%.1f)%n",
                IDX_DIST_LEFT, W_BOUNDARY);
        System.out.printf("  [%d]      distance to right boundary-> boundary penalty (W=%.1f)%n",
                IDX_DIST_RIGHT, W_BOUNDARY);
        System.out.printf("  [%d]     distance to nearest agent  -> proximity penalty (W=%.1f)%n",
                IDX_DIST_AGENT0, W_PROXIMITY);
        System.out.printf("  [%d]     distance to 2nd agent      -> proximity penalty (W=%.1f)%n",
                IDX_DIST_AGENT1, W_PROXIMITY);
        System.out.printf("  collision proxy threshold: %.2f (hard penalty W=%.1f)%n",
                COLLISION_DIST, W_COLLISION);
        System.out.printf("  server obs_dim reported: %d%n%n", obsDim);
        if (obsDim != 32) {
            System.out.println("  WARNING: obs_dim != 32 — index map above assumes default config.");
            System.out.println("  Check is_obs_steering / n_nearing_agents_observed if obs_dim differs.");
            System.out.println();
        }
    }

    // ── Neural network ────────────────────────────────────────────────────
    //
    // Single hidden layer, tanh activations, output scaled to action bounds.
    // Identical architecture to ExampleEvolutionaryRobotics.Network.
    // Students can swap this for any Java model — the ES loop is unchanged.

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
            return hiddenDim * inputDim    // W1
                 + hiddenDim              // b1
                 + outputDim * hiddenDim  // W2
                 + outputDim;             // b2
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
                float t = (float) Math.tanh(sum);  // in (-1, 1)
                out[o] = outLow[o] + (t + 1f) * 0.5f * (outHigh[o] - outLow[o]);
            }
            return out;
        }
    }
}
