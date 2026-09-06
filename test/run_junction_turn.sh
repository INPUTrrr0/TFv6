#!/bin/bash -l
# Does this checkout turn where it is told?
#
#   bash scripts/start_carla.sh &                   # in another shell
#   bash test/run_junction_turn.sh left             # or: right, or both
#
# Runs build -> drive -> check for each direction and exits non-zero on the
# first failure. The three phases keep a red result unambiguous:
#
#   build  finds the junction by id, picks a branch turning the requested way,
#          writes a route through it, and confirms with the leaderboard's own
#          planner that the plan really carries that turn. A failure here is
#          the fixture, not the policy.
#   drive  runs the checkpoint. The agent refuses to start if the plan it is
#          handed does not carry the manoeuvre under test.
#   check  asserts the ego's heading changed the right way between entering
#          and leaving the junction.
#
# Environment:
#   CARLA_PORT    running CARLA's RPC port           (default 2000)
#   CHECKPOINT    checkpoint directory                (default outputs/checkpoints/tfv6_resnet34)
#   TOWN          CARLA town                          (default Town10HD_Opt)
#   JUNCTION      junction id                         (default 189)
#   NO_TRAFFIC    1 to empty the road for a clean demo (default 1)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

PORT="${CARLA_PORT:-2000}"
CHECKPOINT="${CHECKPOINT:-${REPO}/outputs/checkpoints/tfv6_resnet34}"
TOWN="${TOWN:-Town10HD_Opt}"
JUNCTION="${JUNCTION:-189}"
export TFV6_NO_TRAFFIC="${NO_TRAFFIC:-1}"

DIRECTIONS=("$@")
[ ${#DIRECTIONS[@]} -eq 0 ] && DIRECTIONS=(left right)

if [ ! -f "${CHECKPOINT}/config.json" ]; then
    echo "no config.json in ${CHECKPOINT}. Fetch one with:" >&2
    echo "    bash scripts/download_one_checkpoint.sh" >&2
    exit 1
fi

LB="${REPO}/3rd_party/leaderboard"
SR="${REPO}/3rd_party/scenario_runner"
export PYTHONPATH="${REPO}:${LB}:${SR}:${PYTHONPATH:-}"
export LEADERBOARD_ROOT="${LB}"
export SCENARIO_RUNNER_ROOT="${SR}"
export PYTHONUNBUFFERED=1
mkdir -p test/routes test/outputs test/demo_video

status=0
for turn in "${DIRECTIONS[@]}"; do
    name="j${JUNCTION}_${turn}"
    route="test/routes/${name}.xml"
    run_dir="test/outputs/${name}"
    rm -rf "${run_dir}"; mkdir -p "${run_dir}"

    export BENCHMARK_ROUTE_ID="${name}"
    export EVALUATION_OUTPUT_DIR="${run_dir}"
    export SAVE_PATH="${run_dir}"
    export IS_BENCH2DRIVE=0
    export PLANNER_TYPE="${PLANNER_TYPE:-only_traj}"
    export NAV_PROBE_LOG="${run_dir}/nav_probe.jsonl"
    export TFV6_REQUIRE_COMMAND="${turn}"
    # The heuristics upstream's README names as required to reproduce its
    # numbers. Creeping matters most here: without it a policy that stops stays
    # stopped, and ActorBlockedTest fails the route after 180 s with no
    # red-light exemption.
    export LEAD_CLOSED_LOOP_CONFIG="${LEAD_CLOSED_LOOP_CONFIG:-sensor_agent_creeping=True sensor_agent_stuck_threshold=500 use_kalman_filter=True slower_for_stop_sign=True produce_demo_video=true}"

    echo; echo "########## junction ${JUNCTION}, ${turn} turn ##########"
    python test/junction_turn_test.py build --port "${PORT}" --town "${TOWN}" \
        --junction "${JUNCTION}" --turn "${turn}" --out "${route}" || { status=1; continue; }

    python "${LB}/leaderboard/leaderboard_evaluator.py" \
        --routes "${route}" --track SENSORS \
        --checkpoint "${run_dir}/checkpoint_endpoint.json" \
        --agent test/agent.py --agent-config "${CHECKPOINT}" \
        --debug 0 --record None --resume False \
        --port "${PORT}" --traffic-manager-port "$((PORT + 6000))" \
        --traffic-manager-seed 0 --repetitions 1 --timeout 180
    [ $? -ne 0 ] && status=1

    python test/junction_turn_test.py check --junction "${JUNCTION}" \
        --turn "${turn}" --trace "${run_dir}/nav_probe.jsonl" || status=1

    src="$(ls "${run_dir}"/*_demo.mp4 2>/dev/null | head -1)"
    if [ -n "${src}" ] && command -v ffmpeg >/dev/null; then
        ffmpeg -loglevel error -y -i "${src}" -an -c:v libx264 -crf 30 \
            -preset veryfast -vf "scale=iw/2:ih/2" \
            "test/demo_video/junction${JUNCTION}_${turn}.mp4"
        echo "[video] test/demo_video/junction${JUNCTION}_${turn}.mp4"
    fi
done
exit "${status}"
