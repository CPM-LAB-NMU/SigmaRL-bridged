# Simulator vs. Physical CPM Lab

This document describes how the SigmaRL simulator relates to the real CPM Lab hardware, what the differences are, and how student code written against the simulator can eventually be deployed to real vehicles.

---

## The CPM Lab

The CPM (Cyber-Physical Mobility) Lab at RWTH Aachen is a miniature autonomous driving testbed. It consists of:

- A flat track with road markings and intersections, roughly 4.5 m × 4.0 m
- A fleet of small electric vehicles (scale ~1:18), each running an embedded controller
- An **Indoor Positioning System (IPS)** — an overhead camera array that localises every vehicle at ~50 Hz and publishes their pose over the network
- A **ROS2** middleware layer through which all components communicate

### Communication flow in the real lab

```
IPS (overhead cameras)
  │  publishes VehicleStateList at ~50 Hz
  │  fields per vehicle: vehicle_id, x [m], y [m], heading [rad]
  ▼
Student high-level controller
  │  reads VehicleStateList
  │  computes a command per vehicle
  │  publishes VehicleCommandList
  │  fields per command: vehicle_id, speed [m/s], curvature [1/m]
  ▼
Vehicle embedded controller (rosbridge)
  │  converts speed+curvature to motor PWM
  │  executes on the physical car
```

Students never talk to the vehicles directly — they receive state from the IPS and send commands back, and the lab infrastructure handles everything else.

---

## The simulator

SigmaRL uses **VMAS** (Vectorised Multi-Agent Simulator), a differentiable 2D physics simulator running in PyTorch. The `ScenarioRoadTraffic` scenario models the same intersection maps used in the real lab.

The gRPC bridge exposes two interfaces:

### ML interface (`Reset` / `Step`)

Used for training. The simulator returns a **rich observation vector** per agent (`obs_dim` floats) that encodes path deviation, heading error, look-ahead waypoints, and neighbour positions. The action is `[speed, steering_angle]` directly as VMAS internal units.

This interface is purpose-built for learning algorithms — the observation is information-dense and the reward signal is shaped to guide the agent towards good lane-following and collision avoidance behaviour.

### Physical interface (`ResetPhysical` / `StepPhysical`)

Designed to mirror the real lab communication protocol. The simulator returns per-vehicle state in the **lab coordinate frame** (same origin, axes, and units as the IPS), and accepts commands in the **speed + curvature format** used by the real lab.

```
ResetPhysical / StepPhysical
  ├── returns: VehicleStateMsg { vehicle_id, x [m], y [m], heading [rad], speed [m/s] }
  └── accepts: VehicleCommandMsg { vehicle_id, speed [m/s], curvature [1/m] }
```

A controller written against `StepPhysical` processes the same message fields, in the same coordinate frame, as one written for the real lab.

---

## Key differences

### 1. Observation content

| | Physical lab / `StepPhysical` | `Reset` / `Step` (ML interface) |
|---|---|---|
| Position | x, y in lab frame | Encoded in obs vector |
| Heading | Absolute yaw in lab frame | Heading error relative to lane |
| Speed | Scalar speed [m/s] | Encoded in obs vector |
| Lane reference | Not provided — student must compute | Included (look-ahead waypoints) |
| Other vehicles | Not provided — student must fuse | Included (relative positions/velocities) |
| Reward | Not provided | Provided per timestep |

The physical interface gives you raw sensor data, just like the real lab. The ML interface gives you pre-processed features tailored for learning.

### 2. Command format

The physical lab and `StepPhysical` both use **speed + curvature**:

- `speed` [m/s] — desired forward speed
- `curvature` [1/m] — path curvature; positive = left turn

This is related to steering angle by the kinematic bicycle model:

```
steering_angle = atan(curvature × wheelbase)
```

where `wheelbase = 0.15 m` for the CPM miniature vehicles.

The ML interface (`Step`) uses **speed + steering angle** internally (VMAS native format). The bridge converts curvature to steering angle automatically when using `StepPhysical`.

### 3. Coordinate frame

The IPS and `StepPhysical` share the same lab frame:

- Origin at the bottom-left corner of the track
- x axis pointing right (East), y axis pointing up (North)
- Heading: 0 rad = facing East, increases counter-clockwise

VMAS uses a rotated internal frame. The bridge applies a 180° rotation and a fixed shift to convert between them transparently:

```
pos_lab  = (pos_vmas − [world_x_dim, world_y_dim]) × R
heading_lab = heading_vmas − π
```

where `R` is a 180° rotation matrix. Students using `StepPhysical` never see the VMAS frame.

### 4. Timestep and timing

| | Simulator | Physical lab |
|---|---|---|
| Timestep duration | ~0.05 s (fixed) | ~0.02 s (IPS at 50 Hz) |
| Real-time | No — runs as fast as possible | Yes — wall-clock driven |
| Parallelism | `n_envs` copies simultaneously | One physical environment |

In the simulator, time is purely logical. A single `Step()` call advances all `n_envs` environments by one timestep; the wall-clock duration depends on hardware and is typically much less than 50 ms. This means training can run hundreds of episodes per minute.

### 5. Noise and model fidelity

| | Simulator | Physical lab |
|---|---|---|
| Position noise | None | IPS localisation noise (~1–2 cm) |
| Actuator delay | None | ~1 frame latency from command to execution |
| Slipping / wheel dynamics | Not modelled | Present on real surface |
| Battery / motor variation | None | Vehicle-to-vehicle variation |

Controllers that work perfectly in simulation may need robustness tuning before they work well on real hardware. This is the standard **sim-to-real gap**. Increasing observation noise and adding action delay during training can help bridge this gap.

---

## Transfer path: simulator → real lab

A student who has trained a controller against the ML interface and wants to deploy it to real hardware would follow these steps:

1. **Keep the neural network in Java** — no change needed. The weights are just a float array.

2. **Replace the observation source.** In sim, observations come from `SigmaRLClient.agentObs(state, env, agent)`. In the real lab, the equivalent observation vector must be computed from raw IPS data using the same pre-processing SigmaRL applies internally (path deviation, heading error, look-ahead, etc.).

3. **Replace the action sink.** In sim, actions go to `client.step(actions)`. In the real lab, they are published as `VehicleCommandList` messages over ROS2.

Alternatively, a student can develop a controller directly against the **physical interface** (`StepPhysical`). This controller processes the same `[x, y, heading, speed]` fields as the real IPS and sends the same speed+curvature commands. Only the transport layer (gRPC vs. ROS2) needs to change for real deployment. However, such a controller must compute all path and collision avoidance logic from scratch, since the raw physical state contains no reference information.

---

## Summary

| Aspect | Physical lab | `StepPhysical` | `Reset` / `Step` |
|--------|-------------|----------------|-----------------|
| State format | `VehicleStateList` (ROS2) | `VehicleStateMsg` (gRPC) | flat float vector (gRPC) |
| Command format | `VehicleCommand` speed+curv (ROS2) | speed+curvature (gRPC) | speed+steering (gRPC) |
| Coordinate frame | Lab frame | Lab frame | VMAS internal (hidden) |
| Reference path provided | No | No | Yes (in obs vector) |
| Suitable for learning | Manual feature engineering required | Manual feature engineering required | Yes — designed for it |
| Suitable for real deployment | Native | Direct transfer (change transport only) | Requires obs reconstruction |
| Runs in real time | Yes | No (faster) | No (faster) |
