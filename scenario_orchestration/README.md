# TransFuser v6 (`lead`) as an `ego_policy_v1` ego policy

This directory is the whole integration. It adds two files to an otherwise
unmodified checkout of [`lead`](https://github.com/kesai-labs/lead) so a
scenario harness can drive the released checkpoint without knowing anything
about sensor fusion, jaxtyping batches, or CARLA blueprint attributes.

```text
lead/                           # upstream, unmodified
└── scenario_orchestration/
    ├── policy.py               # the adapter — the only code we add
    └── README.md               # this file
```

The harness calls `build_policy(policy_json)` once and `policy.act(observation)`
per decision. No upstream file is touched, so `git pull upstream main` stays a
fast-forward.

## Quick start

```bash
export AV_CKPT=/path/to/checkpoints     # see "Checkpoint"
export CARLA_ROOT=/path/to/CARLA_0.9.16
# `lead` subclasses the CARLA leaderboard's AutonomousAgent, so leaderboard and
# scenario_runner must import even though no leaderboard runs:
export PYTHONPATH="$PWD:$PWD/3rd_party/leaderboard/standard/leaderboard:$PWD/3rd_party/leaderboard/standard/scenario_runner:$PYTHONPATH"
```

From the highway orchestrator it is a one-word argument:

```bash
./run_record_policy.sh tfv6 cutin
```

## Configuration

| key | default | what it does |
|---|---|---|
| `checkpoint` | `$AV_CKPT/tfv6/resnet34_v1.5.0/seed0` | the released run directory |
| `parameters.device` | `cuda:0` | a GPU is required |
| `parameters.town` | `Town05` | passed through to `lead`'s own map-dependent setup |
| `TFV6_RADAR` (env) | `sensor` | `zeros` restores an all-zero radar tensor, for an A/B |
| `TFV6_LEADERBOARD` (env) | `standard` | which vendored leaderboard variant to import |

## The sensor rig — get this wrong and it drives into things

The adapter declares its rig through `sensors()`, and **every field is taken
from `lead`'s own `config/expert/sensor_rig_config.py`, not chosen here.** This
is the single most load-bearing part of the file, because a wrong rig fails
silently: the model still runs, still returns controls, and simply drives badly.

| sensor | size / spec | fov | yaw | mount `(x, y, z)` |
|---|---|---|---|---|
| `PCAM_L0` | 384×384 | 60° | −57.5° | `(0.0, −0.3, 2.25)` |
| `PCAM_F0` | 384×384 | 60° | 0° | `(0.25, 0.0, 2.25)` |
| `PCAM_R0` | 384×384 | 60° | +57.5° | `(0.0, +0.3, 2.25)` |
| `lidar` | 64 ch, 100 m | — | — | `(1.0, 0.0, 2.5)` |
| 4 × radar | per `SensorRigConfig.radars` | — | — | — |

An earlier version of this rig used `(-1.5, 0, 2.0)`, 90° FOV and ±60° yaw —
wrong on *every* axis. On the leaderboard hero (`vehicle.lincoln.mkz_2020`) that
puts the lens **inside the cabin**, and the policy targeted roughly 0 m/s on
open road. Correcting the rig took the same checkpoint from 3.4 m travelled in
15 s to 95 m with zero collisions. If you port this adapter elsewhere, diff the
rig against `SensorRigConfig` rather than trusting these numbers.

**LiDAR timing is not decorative either.** `rotation_frequency` is a
revolutions-per-second figure, so in a 60 Hz world a stock `rotation_frequency=20`
sweeps one third of a revolution per tick — a fixed 180–299° wedge that never
faces forward. The port's `carla_sensors.for_tick_rate()` retimes rotation and
`points_per_second` against the world's actual `fixed_delta_seconds`; a port
without that must do the same.

## What the policy returns

```python
{
  "control":   {"throttle": float, "steer": float, "brake": float},
  "waypoints": [[x, y], ...],
  "meta":      {"policy": "tfv6", "step": int, ...},
}
```

Controls come from `lead`'s own `WaypointTracker` / `PathSpeedTracker` through
the modality selection `TransfuserAgent` applies — not re-derived here.

## Navigation

TFv6 has **no command input**. `TransfuserForwardBatch` carries `rgb`,
`rasterized_lidar`, `radar`, `route`, `previous_target_point`, `target_point`,
`next_target_point` and `speed` — and nothing else. `RoadOption.CHANGELANELEFT`
appears in the repository only inside `src/lead/expert/`, the privileged expert
that generated training data; the network never saw it.

So navigation reaches this policy as **geometry only**. A turn or a lane change
is expressed by target points that sit where you want the car to go. There is no
symbolic or language channel to address, and no way to "tell it" to turn.

## Checkpoint

```text
$AV_CKPT/tfv6/resnet34_v1.5.0/seed0/     # <- `checkpoint` points here
```

Keep weights outside this checkout: a download inside the tree shows up forever
as uncommitted changes in any parent repository that vendors this one.
