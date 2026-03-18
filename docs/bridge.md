# SigmaRL gRPC Bridge

A language-agnostic training backend that exposes the SigmaRL/VMAS simulator over gRPC, allowing neural networks and evolutionary algorithms written in **any language** to train against the full SigmaRL environment.

Java is the primary target, but the same server works for Python, C++, Rust, or any language with gRPC support.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
  - [Python server](#python-server)
  - [Java client](#java-client)
- [Quick Start](#quick-start)
  - [Evolutionary Algorithm (EA) mode](#evolutionary-algorithm-ea-mode)
  - [Step-by-step NN training mode](#step-by-step-nn-training-mode)
- [gRPC Service Reference](#grpc-service-reference)
  - [GetSpaces](#getspaces)
  - [GetWeightCount](#getweightcount)
  - [Configure](#configure)
  - [Reset](#reset)
  - [Step](#step)
  - [EvaluateWeights](#evaluateweights)
- [Message Reference](#message-reference)
- [Array Layout](#array-layout)
- [Extending the Bridge](#extending-the-bridge)
  - [Adding a new field to the proto](#adding-a-new-field-to-the-proto)
  - [Adding a new language client](#adding-a-new-language-client)
- [Design Decisions](#design-decisions)

---

## Motivation

SigmaRL trains multi-agent RL policies in Python using PyTorch and VMAS. The deployment target is the CPM Lab (ROS2-based), but the training pipeline has no ROS2 dependency.

The goal of this bridge is to let a supervisor define **neural network architectures** and **evolutionary algorithms** in Java (or any other language) while reusing the full SigmaRL simulation stack — scenarios, observations, reward shaping, CBF safety filters, and vectorised parallel environments — without reimplementing any of it.

### Why gRPC and not rosbridge?

rosbridge is designed for ROS2 topic communication. The training pipeline does not use ROS2 at all; adding it just to bridge a training client would introduce a large, unnecessary dependency. rosbridge remains the right tool for the *deployment* side (connecting trained policies to the ROS2 lab stack), but for training it is the wrong abstraction.

gRPC was chosen because:

- Strong typing via protobuf (not stringly-typed JSON)
- Efficient binary serialisation of float arrays (weight vectors, observations, actions)
- First-class Java support via `grpc-java`
- Trivially extensible to any language that has a gRPC implementation
- The same server can later coexist with a rosbridge deployment node without conflict

---

## Architecture

```
┌───────────────────────────────────────────────┐
│               Java (or any language)          │
│                                               │
│  Mode A — Evolutionary Algorithm              │
│    population = [w1, w2, ..., wN]             │
│    fitness = EvaluateWeights(wi, n_episodes)  │
│    // Java owns selection, mutation, etc.     │
│                                               │
│  Mode B — Step-by-step NN training            │
│    obs  = Reset()                             │
│    action = myNet.forward(obs)                │
│    obs, reward, done = Step(action)           │
│    // Java owns backprop / policy gradient    │
└──────────────────┬────────────────────────────┘
                   │  gRPC over TCP (protobuf)
┌──────────────────┴────────────────────────────┐
│          SigmaRL gRPC Server  (Python)        │
│                                               │
│  sigmarl/bridge/server.py                     │
│    ├─ GetSpaces / GetWeightCount              │
│    ├─ Configure                               │
│    ├─ Reset / Step                            │
│    └─ EvaluateWeights                        │
│                                               │
│  sigmarl/bridge/env_adapter.py               │
│    └─ wraps VmasEnv + TransformedEnvCustom   │
│         (ScenarioRoadTraffic, rewards, CBF)  │
└───────────────────────────────────────────────┘
```

**Java owns the learning algorithm.** Python owns the simulator. The boundary is clean: Java sends actions (or weight vectors), Python returns observations and rewards.

---

## Directory Structure

```
SigmaRL-bridged/
├── sigmarl/bridge/
│   ├── sigmarl_env.proto          # Single source of truth for the API contract
│   ├── sigmarl_env_pb2.py         # Generated — do not edit by hand
│   ├── sigmarl_env_pb2_grpc.py    # Generated — do not edit by hand
│   ├── env_adapter.py             # VMAS wrapper with flat float-list API
│   └── server.py                  # gRPC servicer entry point
│
├── java-client/
│   ├── pom.xml                    # Maven build; generates Java stubs at compile time
│   └── src/main/java/io/sigmarl/bridge/
│       ├── SigmaRLClient.java     # Clean Java wrapper around the generated stub
│       └── example/
│           ├── ExampleEA.java         # Genetic algorithm using EvaluateWeights
│           └── ExampleNNTrainer.java  # Step-by-step loop with placeholder network
│
├── requirements_bridge.txt        # grpcio, grpcio-tools, protobuf
└── scripts/generate_proto.sh      # Regenerates Python stubs from proto
```

---

## Setup

### Python server

1. **Install bridge dependencies** (separate from core SigmaRL deps):

   ```bash
   conda activate sigmarl
   pip install -r requirements_bridge.txt
   ```

2. **Python stubs are already committed** to `sigmarl/bridge/`. If you modify the proto, regenerate them:

   ```bash
   bash scripts/generate_proto.sh
   ```

3. **Start the server:**

   ```bash
   python -m sigmarl.bridge.server          # port 50051 (default)
   python -m sigmarl.bridge.server 50052    # custom port
   ```

### Java client

Requires Java 17+ and Maven 3.8+.

```bash
cd java-client
mvn package          # downloads grpc-java, compiles proto, builds fat jar
```

The proto file is read directly from `../sigmarl/bridge/sigmarl_env.proto` — there is a single copy shared between both builds.

---

## Quick Start

### Evolutionary Algorithm (EA) mode

Java manages a population of weight vectors. Python evaluates fitness by running full episodes.

```java
try (SigmaRLClient client = new SigmaRLClient("localhost", 50051)) {

    // Optional: configure the scenario
    client.configure(SigmaRLClient.scenarioConfig(
        "intersection_1", /*n_agents=*/4, /*n_envs=*/4, /*max_steps=*/128, "cpu"));

    long nWeights = client.getWeightCount();  // size of each weight vector

    // Evaluate a single individual
    float[] weights = new float[(int) nWeights];  // your EA fills this
    EvaluateResponse resp = client.evaluateWeights(weights, /*n_episodes=*/5);

    System.out.println(resp.getMeanReward());
}
```

See `ExampleEA.java` for a complete genetic algorithm (tournament selection + Gaussian mutation).

### Step-by-step NN training mode

Java drives the episode loop. Useful when backpropagation happens on the Java side (e.g. with DL4J).

```java
try (SigmaRLClient client = new SigmaRLClient("localhost", 50051)) {

    SpacesInfo spaces = client.getSpaces();
    int nAgents   = spaces.getNAgents();
    int actionDim = spaces.getActionDim();

    StepResponse state = client.reset();

    while (!SigmaRLClient.anyDone(state)) {
        float[] actions = new float[nAgents * actionDim];
        // ... fill actions from your network's forward pass ...

        StepResponse next = client.step(actions);

        // ... compute loss, call backward(), update weights ...

        state = next;
    }
}
```

See `ExampleNNTrainer.java` for a full skeleton with a placeholder single-hidden-layer network.

---

## gRPC Service Reference

The full service definition is in `sigmarl/bridge/sigmarl_env.proto`.

### GetSpaces

Returns the shapes and action bounds of the currently configured environment. Call this after `Configure` (or on startup if using the defaults) to size your network inputs and outputs.

```
GetSpaces(Empty) → SpacesInfo
```

| Field | Type | Description |
|-------|------|-------------|
| `n_agents` | int32 | Number of agents |
| `n_envs` | int32 | Number of parallel VMAS environments |
| `obs_dim` | int32 | Per-agent observation size |
| `action_dim` | int32 | Per-agent action size |
| `action_low` | float[] | Lower bounds, length = `action_dim` |
| `action_high` | float[] | Upper bounds, length = `action_dim` |

### GetWeightCount

Returns the total number of scalar parameters in the default policy architecture (3-layer MLP, 256 hidden units, parameter-shared across agents). This is the required length of weight vectors passed to `EvaluateWeights`.

```
GetWeightCount(Empty) → WeightCountInfo { n_weights: int64 }
```

### Configure

Reconfigures the scenario. Must be called before `Reset`/`Step` if the defaults are not suitable. Rebuilds the environment, so it resets state.

```
Configure(ScenarioConfig) → Ack
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scenario_type` | string | `intersection_1` | Key from `sigmarl/constants.py` SCENARIOS dict |
| `n_agents` | int32 | 4 | Overridden by SCENARIOS if scenario defines a fixed count |
| `n_envs` | int32 | 1 | VMAS parallel envs — increase for faster EA evaluation |
| `max_steps` | int32 | 128 | Episode length |
| `device` | string | `cpu` | `"cpu"` or `"cuda"` |
| `random_seed` | int32 | 0 | |
| `is_partial_observation` | bool | true | |
| `is_ego_view` | bool | true | |

Available scenario types (from `sigmarl/constants.py`):
`CPM_entire`, `CPM_mixed`, `intersection_1`, `interchange_1`, `on_ramp`, `roundabout`, and others.

### Reset

Resets all parallel environments and returns initial observations.

```
Reset(ResetRequest { seed: int32 }) → StepResponse
```

`seed = 0` uses the seed set in `Configure`. Any non-zero value overrides it for this reset only.

`rewards` and `dones` in the response are zero/false — there are no rewards on the first step.

### Step

Advances all parallel environments by one timestep.

```
Step(StepRequest { actions: float[] }) → StepResponse
```

`actions` must be a flat row-major array of shape `[n_envs × n_agents × action_dim]`. Values are clipped to `[action_low, action_high]` internally by VMAS.

### EvaluateWeights

Loads a flat weight vector into the default policy architecture, runs `n_episodes` complete episodes, and returns the cumulative rewards. This is the EA fitness oracle.

```
EvaluateWeights(EvaluateRequest) → EvaluateResponse
```

| Request field | Description |
|---------------|-------------|
| `weights` | Flat float array, length = `GetWeightCount()` |
| `n_episodes` | Episodes to run (averaged for `mean_reward`) |
| `config` | Optional `ScenarioConfig` override for this call only |

| Response field | Description |
|----------------|-------------|
| `episode_rewards` | Per-episode cumulative reward (summed over agents + timesteps) |
| `mean_reward` | Mean of `episode_rewards` |
| `std_reward` | Std dev of `episode_rewards` |

---

## Message Reference

### StepResponse

The common response for both `Reset` and `Step`.

| Field | Shape | Description |
|-------|-------|-------------|
| `observations` | `[n_envs × n_agents × obs_dim]` | Flat row-major |
| `rewards` | `[n_envs × n_agents]` | Flat row-major |
| `dones` | `[n_envs]` | True if that environment's episode ended |
| `n_envs` | scalar | Convenience: same as `SpacesInfo.n_envs` |
| `n_agents` | scalar | Convenience: same as `SpacesInfo.n_agents` |
| `obs_dim` | scalar | Convenience: same as `SpacesInfo.obs_dim` |

---

## Array Layout

All multi-dimensional arrays are **flat and row-major**. The indexing convention:

```
observations[env_i, agent_j, obs_k]  →  flat index: (env_i * n_agents + agent_j) * obs_dim + obs_k
rewards[env_i, agent_j]              →  flat index: env_i * n_agents + agent_j
actions[env_i, agent_j, action_k]   →  flat index: (env_i * n_agents + agent_j) * action_dim + action_k
```

`SigmaRLClient` provides helpers:

```java
float[] obs    = SigmaRLClient.agentObs(resp, envIdx, agentIdx);
float   reward = SigmaRLClient.agentReward(resp, envIdx, agentIdx);
boolean done   = SigmaRLClient.anyDone(resp);
```

---

## Extending the Bridge

### Adding a new field to the proto

1. Edit `sigmarl/bridge/sigmarl_env.proto`.
2. Regenerate Python stubs:
   ```bash
   bash scripts/generate_proto.sh
   ```
3. Java stubs are regenerated automatically on the next `mvn package`.
4. Update `env_adapter.py` and/or `server.py` to handle the new field.

Proto3 is backwards-compatible for additions — existing clients ignore unknown fields.

### Adding a new language client

Any language with a gRPC implementation can connect to the server. The general steps are:

1. Copy or symlink `sigmarl/bridge/sigmarl_env.proto` into your project.
2. Use that language's protoc plugin to generate stubs.
3. Open a channel to `localhost:50051` and call the service.

Official gRPC support exists for: C++, C#, Dart, Go, Java, Kotlin, Node.js, Objective-C, PHP, Python, Ruby, Rust (community).

---

## Design Decisions

**Single proto file, two builds.** `sigmarl/bridge/sigmarl_env.proto` is the single source of truth. The Maven `pom.xml` is configured with `<protoSourceRoot>` pointing to that file so there is no duplication.

**`env_adapter.py` has no gRPC imports.** The VMAS wrapping logic is fully decoupled from the transport layer, making it straightforward to add alternative transports (e.g. ZeroMQ, shared memory) or to test the adapter in isolation.

**Generated Python stubs are committed.** This avoids requiring `grpcio-tools` for users who only run the server (they only need `grpcio`). The stubs are small and deterministic.

**`n_envs > 1` for EA.** VMAS is designed for vectorised simulation. Setting `n_envs` to 4–32 in `Configure` means `EvaluateWeights` runs that many episodes in parallel internally, dramatically reducing wall-clock time per fitness evaluation at no extra cost to the Java caller.
