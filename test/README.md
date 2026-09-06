# `test/` — does this checkout turn where it is told?

Run this before building anything on top of this repository. It drives the
released checkpoint through one junction in each direction and asserts, from the
ego's own heading, that the turn happened. If it does not reproduce
`demo_video/`, something in your environment differs from the one these
recordings came from, and every number you take afterwards inherits that.

```bash
bash scripts/download_one_checkpoint.sh     # once
bash scripts/start_carla.sh &               # in another shell
bash test/run_junction_turn.sh              # both directions
```

## What it should produce

Town10HD_Opt, junction 189, an empty road:

| | driving score | route completion | heading change inside the junction |
| :--- | ---: | ---: | ---: |
| left | 100.0 | 100 | **-64.2 deg** |
| right | 100.0 | 100 | **+89.8 deg** |

CARLA's yaw grows clockwise seen from above, so a negative change is a left turn
and a positive one a right turn. `demo_video/junction189_left.mp4` and
`demo_video/junction189_right.mp4` are those two runs.

Nothing about the route is hand-placed: the input is a junction id and a
direction, and `junction_turn_test.py` enumerates the junction's manoeuvres,
keeps the ones turning the requested way, and searches them against several
route lengths until the leaderboard's own planner returns a plan that both
carries the turn and stays within twice the direct distance.

## Three phases, so a red result is never ambiguous

| phase | what it establishes | a failure here means |
| :--- | :--- | :--- |
| **build** | the written route really plans as the requested turn, checked with the leaderboard's own `interpolate_trajectory` | the fixture is wrong; nothing was driven |
| **drive** | the checkpoint drives it; the agent refuses to start if the plan it receives lacks the manoeuvre | the environment or the checkpoint |
| **check** | the ego's heading changed the right way between entering and leaving the junction | the policy |

The check deliberately does not correlate the command with a steering peak. That
offset was measured at +258 steps for the right turn and -103 for the left, so
any window over it either misses the manoeuvre or matches an unrelated lane
adjustment — which produced a false pass before it was replaced.

## Things worth knowing before you read the numbers

**`Perfect` here means an empty road.** `NO_TRAFFIC=1` is the default for this
test and removes the other vehicles every ten ticks, because the leaderboard
adds `BackgroundBehavior` unconditionally and offers no flag for a clear world.
It isolates the manoeuvre; it is not a score. Set `NO_TRAFFIC=0` for traffic,
and expect a lower one.

**The heuristics are on.** `sensor_agent_creeping`, `use_kalman_filter` and
`slower_for_stop_sign` all default to `False` in `config_closed_loop.py`, and
the README names them as required to reproduce the published numbers. Without
creeping, a policy that stops stays stopped and `ActorBlockedTest` fails the
route after 180 s — with no red-light exemption. The creep threshold is set to
500 frames here rather than 1100 so a stall is broken well before that limit.

**The GNSS projection is checked, not assumed.** CARLA moved its GNSS sensor
from an equatorial to a transverse Mercator in 0.9.16, and this branch
implements only the 0.9.15 inverse. `test/agent.py` applies the 0.9.16 inverse
when the server reports 0.9.16, leaves it alone on 0.9.15, and refuses to guess
for anything else — a source build reporting a git hash must say which it is
through `TFV6_GNSS_TRANSVERSE_MERCATOR`. Decoding with the wrong projection
mislocalizes the ego silently.

## Knobs

| | |
| :--- | :--- |
| `CARLA_PORT` | RPC port of the running server (2000) |
| `CHECKPOINT` | checkpoint directory (`outputs/checkpoints/tfv6_resnet34`) |
| `TOWN` / `JUNCTION` | which intersection (`Town10HD_Opt` / `189`) |
| `NO_TRAFFIC` | `0` to keep the background traffic (`1`) |
| `TFV6_GNSS_TRANSVERSE_MERCATOR` | force the projection on an unrecognised server |
