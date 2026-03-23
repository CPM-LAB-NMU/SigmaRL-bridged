# Training with the gRPC Bridge

This document explains how the Java example programs train neural network controllers using evolutionary algorithms against the SigmaRL simulator.

---

## Overview

Training follows the standard **neuroevolution** loop: a population of neural networks is evaluated by running episodes in the simulator, the best-performing networks are selected as parents, and the next generation is produced by mutating (and optionally recombining) the parents' weights.

**Java owns the entire learning algorithm.** The Python server is a pure black-box simulator: Java sends actions, Python advances the physics and returns the next observation and reward. Java never calls PyTorch or imports any ML library.

---

## What the simulator provides

Before writing any training code, call `GetSpaces` to learn the dimensions of the problem:

```java
SpacesInfo spaces = client.getSpaces();
int obsDim    = spaces.getObsDim();     // size of each agent's observation vector
int actionDim = spaces.getActionDim();  // number of outputs your network must produce
float[] actionLow  = ...spaces.getActionLowList();
float[] actionHigh = ...spaces.getActionHighList();
```

These values are fixed for a given scenario and configuration, and are all you need to construct the network.

### The observation vector

Each agent receives a vector of `obsDim` floats from the simulator at every timestep. This vector is designed by SigmaRL specifically to be learnable — it encodes everything the agent needs to make a good driving decision:

- **Path deviation** — lateral distance from the centre of the agent's assigned lane
- **Heading error** — angular difference between the agent's current heading and the lane direction
- **Look-ahead waypoints** — a sequence of upcoming reference path points ahead of the agent
- **Neighbouring agent positions and velocities** — relative positions and speeds of nearby agents, allowing the network to reason about collisions
- **Speed** — the agent's current scalar speed

This is why training works: the observation vector contains the goal (stay on this lane, face this direction) and the context (other cars nearby). A network that learns to minimise path deviation and heading error while keeping a safe distance will naturally exhibit good behaviour.

Contrast this with the physical state `[x, y, heading, speed]`, which has no reference to where the agent *should* be — the network cannot infer its goal from those four numbers alone.

### The action vector

Each agent produces `actionDim` floats (typically 2):

| Index | Meaning | Typical bounds |
|-------|---------|----------------|
| 0 | Target speed [m/s] | `[-0.5, 1.0]` |
| 1 | Steering angle [rad] | `[-0.6, 0.6]` |

The exact bounds are returned by `getActionLow()` / `getActionHigh()` and depend on the configured scenario. Always read these at runtime rather than hardcoding them.

### The reward signal

`StepResponse.rewards` contains one reward per `(env, agent)` pair. SigmaRL's reward is a shaped scalar that combines:

- Progress along the reference path (positive)
- Proximity to other agents / collision penalty (negative)
- Deviation from the lane centre (negative)

This signal is dense and informative — it fires every timestep, so even short episodes contain a useful learning gradient.

---

## Episode lifecycle

A single training episode from the Java side looks like this:

```
Reset()  →  initial StepResponse (obs filled, rewards = 0, dones = false)
Step()   →  StepResponse  (obs, rewards, dones)
Step()   →  StepResponse
  ...
Step()   →  StepResponse  (dones[0] = true  ← episode over)
```

An episode ends when:
- **`max_steps` is reached** — the configured episode length (default 128 steps)
- **A terminal event occurs** — collision, agent leaving the map, etc.

Both cases set `dones[env_index] = true` in the response. In Java:

```java
while (!SigmaRLClient.anyDone(state)) {
    state = client.step(actions);
}
```

After the loop, call `Reset()` to start a new episode.

---

## Neural network architecture

Both examples use a **single hidden layer feedforward network** (fully connected, tanh activations):

```
input layer   : obs_dim neurons   — one per element of the observation vector
hidden layer  : HIDDEN_DIM neurons — tanh activation
output layer  : action_dim neurons — scaled to [action_low, action_high]
```

Output scaling maps tanh's range `(-1, 1)` to the physical action bounds:

```java
float t = (float) Math.tanh(preactivation);
out[o] = outLow[o] + (t + 1f) * 0.5f * (outHigh[o] - outLow[o]);
```

This ensures the network never produces out-of-range commands regardless of its weights.

The network is parameter-shared across agents: the same weights are used for all four vehicles. This reduces the search space and encourages homogeneous behaviour.

**Weight count** for the default hidden size of 64:

```
W1: obs_dim × 64
b1: 64
W2: 64 × action_dim
b2: action_dim
─────────────────
total ≈ obs_dim × 64 + 64 + 64 × action_dim + action_dim
```

For `obs_dim = 50` and `action_dim = 2`: 50×64 + 64 + 64×2 + 2 = **3394 parameters**.

---

## ExampleEvolutionaryRobotics — (μ, λ)-Evolution Strategy

`ExampleEvolutionaryRobotics.java` implements a **(μ=5, λ=20)-ES**:

### Algorithm outline

```
Initialise: 20 individuals with weights ~ N(0, 0.1²)

For each generation:
  1. Evaluate: run one episode per individual; fitness = cumulative reward
  2. Select:   keep the top μ=5 by fitness (the "parents")
  3. Reproduce: generate λ=20 new individuals by mutating a randomly
                chosen parent: w_new = w_parent + N(0, σ²)
                σ = MUTATION_STD = 0.05
```

All 20 parents are discarded each generation — only the offspring survive. This is "comma selection" `(μ, λ)` as opposed to "plus selection" `(μ + λ)`. It keeps population diversity higher at the cost of not guaranteeing the best individual is preserved.

### Fitness function

```java
totalFitness += SigmaRLClient.agentReward(state, env, a);  // per step, per agent
```

Fitness is the sum of rewards over all agents and all timesteps in the episode. A network that navigates all four agents safely and efficiently for 128 steps will score much higher than one that causes an early collision.

Students can replace this with any scalar derived from `agentReward()` or the observation vectors.

---

## ExampleEA — Genetic Algorithm

`ExampleEA.java` implements a **GA with elite carry-over and uniform crossover**:

### Algorithm outline

```
Initialise: 20 individuals with weights ~ N(0, 0.1²)

For each generation:
  1. Evaluate:   run one episode per individual
  2. Rank:       sort by fitness
  3. Elite:      copy top ELITE_K=4 individuals unchanged to next generation
  4. Offspring:  fill remaining 16 slots:
                   a. Select two parents uniformly at random from the elite
                   b. Uniform crossover: each weight independently from parent 1 or 2
                   c. Gaussian mutation: w += N(0, 0.05²)
```

Elite carry-over guarantees the best solution found so far is never lost.

### Fitness function

```java
// +1 for every step the episode is still running
totalFitness += computeFitness(state);
```

Fitness is **survival time** — the number of timesteps completed before the episode ends. A perfect controller that avoids all collisions and stays on the map for all 128 steps scores 128; one that crashes on step 3 scores 3.

This deliberately decouples the fitness from the simulator's internal reward, demonstrating that students can define any objective. Students are encouraged to replace `computeFitness()` with their own metric.

---

## Configuring a training run

```java
ScenarioConfig cfg = SigmaRLClient.scenarioConfig(
    "intersection_1",   // scenario name
    4,                  // n_agents
    1,                  // n_envs (increase for faster evaluation via parallelism)
    128,                // max_steps per episode
    "cpu"               // device: "cpu" or "cuda"
);
client.configure(cfg);
```

**Increasing `n_envs`** runs that many independent copies of the environment in parallel inside VMAS. With `n_envs=4` the single `Step()` call advances 4 episodes simultaneously, effectively multiplying the data collected per wall-clock second. For the examples, set this to 1 initially; once training is working, try 4–8 for faster convergence.

---

## Rendering the trained controller

Training is always done without rendering (rendering at every step reduces speed by ~10×). After training, the best individual is replayed with the renderer armed:

```java
// Arm the renderer for the replay episode only
client.setRenderMode("human");
runEpisode(client, net, nAgents, nEnvs, actionDim);
client.setRenderMode("");  // disarm

// Or save to video instead:
client.setRenderMode("rgb_array");
runEpisode(client, net, nAgents, nEnvs, actionDim);
client.setRenderMode("");
client.saveVideo("my_best_run");  // writes my_best_run.mp4
```

From the command line:

```bash
java -cp target/sigmarl-java-client-*.jar \
     io.sigmarl.bridge.example.ExampleEvolutionaryRobotics --render

java -cp target/sigmarl-java-client-*.jar \
     io.sigmarl.bridge.example.ExampleEvolutionaryRobotics --save-video output/best
```

---

## Tips for improving results

| What to try | Effect |
|-------------|--------|
| Increase `GENERATIONS` (50 → 100+) | More optimisation time; fitness usually keeps climbing slowly |
| Increase `POPULATION` (20 → 50) | Broader search; helps early on when the landscape is unknown |
| Reduce `MUTATION_STD` (0.05 → 0.02) | Finer local search once a good region is found |
| Increase `HIDDEN_DIM` (64 → 128) | More expressive network; increases weight count and thus search difficulty |
| Increase `n_envs` (1 → 4) | Faster wall-clock time; no change to algorithm |
| Use `n_envs > 1` and average fitness over envs | More stable fitness estimates; reduces noise from random initial conditions |
