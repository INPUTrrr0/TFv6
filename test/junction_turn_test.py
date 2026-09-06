"""Test: TFv6 takes a specified turn at a specified junction.

Two phases, so a failure is attributed to the right half of the system:

  build  — find the junction by id, pick the branch that turns the requested
           way, write a route XML through it, and confirm with the
           leaderboard's own planner that the plan really carries that turn.
           If this fails, the route is wrong and driving it proves nothing.

  check  — read the drive's nav_probe trace and assert the policy actually did
           it: the discrete command reached the model, and the steering went
           the matching way. CARLA's world is left-handed (x forward, y right),
           so a left turn is negative steer and a right turn positive.

    python test/junction_turn_test.py build --port 2000 --town Town10HD_Opt \
        --junction 189 --turn left --out test/routes/j189_left.xml
    python test/junction_turn_test.py check --junction 189 --turn left \
        --trace test/outputs/j189_left/nav_probe.jsonl

`run_junction_turn.sh` runs build -> drive -> check and exits non-zero on the
first failure. See test/README.md.
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET

STRAIGHT_LATERAL_RATIO = 0.35
STEER_THRESHOLD = 0.15

WEATHER = {
    "route_percentage": "0",
    "cloudiness": "10.0",
    "precipitation": "0.0",
    "precipitation_deposits": "0.0",
    "wetness": "0.0",
    "wind_intensity": "5.0",
    "sun_azimuth_angle": "220.0",
    "sun_altitude_angle": "45.0",
    "fog_density": "2.0",
    "fog_distance": "0.75",
}


def classify(entry, exit_wp) -> tuple[str, float, float]:
    """Which way a junction manoeuvre turns, in the entry waypoint's frame."""
    yaw = math.radians(entry.transform.rotation.yaw)
    dx = exit_wp.transform.location.x - entry.transform.location.x
    dy = exit_wp.transform.location.y - entry.transform.location.y
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
    if abs(lateral) < STRAIGHT_LATERAL_RATIO * max(abs(forward), 1e-6):
        return "straight", forward, lateral
    return ("right" if lateral > 0 else "left"), forward, lateral


def build(args) -> int:
    """Write a route through `--junction` that turns `--turn`."""
    import carla

    client = carla.Client("localhost", args.port)
    client.set_timeout(180.0)

    available = [m.split("/")[-1] for m in client.get_available_maps()]
    if args.town not in available:
        print(f"town {args.town!r} not available. Installed: {sorted(available)}")
        return 1
    world = client.load_world(args.town)
    carla_map = world.get_map()

    # Enumerate junctions by sampling the drivable network; carla.Map exposes no
    # direct "junction by id" lookup.
    junctions: dict[int, object] = {}
    for waypoint in carla_map.generate_waypoints(2.0):
        if waypoint.is_junction:
            junction = waypoint.get_junction()
            junctions.setdefault(junction.id, junction)
    if args.junction not in junctions:
        print(f"junction {args.junction} not found on {args.town}.")
        print(f"{len(junctions)} junctions present: {sorted(junctions)}")
        return 1
    junction = junctions[args.junction]
    print(f"junction {args.junction} on {args.town}")

    # Plan with the leaderboard's own entry point rather than reimplementing it:
    # a hand-rolled GlobalRoutePlanner call reported a single LEFT where the plan
    # the agent actually received was a 358-point detour with three RIGHTs.
    from leaderboard.utils.route_manipulation import interpolate_trajectory
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)

    # Every manoeuvre through the junction, from every approach.
    options = []
    for entry, exit_wp in junction.get_waypoints(carla.LaneType.Driving):
        name, forward, lateral = classify(entry, exit_wp)
        options.append((name, entry, exit_wp, forward, lateral))
        print(f"  branch {name:<8} entry road {entry.road_id:>5} lane {entry.lane_id:>3}"
              f"  forward {forward:7.1f}  lateral {lateral:7.1f}")

    matching = [o for o in options if o[0] == args.turn]
    if not matching:
        print(f"junction {args.junction} offers {sorted({o[0] for o in options})}, "
              f"not {args.turn!r}")
        return 1
    # Sharpest first — the manoeuvre is then unambiguous in the trace — but try
    # every branch that turns the right way. A junction offers the same turn
    # from several approach lanes and the planner does not treat them alike: at
    # junction 189 the sharpest right could only be reached by looping the block,
    # 5.6x the direct distance, while other branches may route straight through.
    matching.sort(key=lambda o: abs(o[4]), reverse=True)

    # Keypoints: approach, the junction ENTRY, just past the turn, and a tail.
    # The entry point pins the manoeuvre — given only an approach and an exit the
    # planner is free to reach that exit any way it likes, and on Town10HD_Opt it
    # did, taking 358 m between endpoints 80 m apart.
    #
    # The tail is chosen, not fixed. Its leg is planned like any other, so a tail
    # the planner cannot reach directly turns the route into a loop after the
    # turn — which is how the right-turn route came out at 375 points and drove
    # itself off the plan. Shorter tails are tried until the plan is both
    # carrying the requested turn and reasonably direct; the last candidate has
    # no tail at all, ending the route just past the junction.
    def plan(points):
        """The leaderboard's own plan for these keypoints, and its commands."""
        _, planned = interpolate_trajectory(points)
        names = [option.name for _, option in planned]
        return planned, [
            n for i, n in enumerate(names) if i == 0 or n != names[i - 1]
        ]

    def direct_length(points) -> float:
        return sum(a.distance(b) for a, b in zip(points, points[1:]))

    expected = args.turn.upper()
    positions = None
    for branch_index, (_, entry, exit_wp, _, lateral) in enumerate(matching):
        back = entry.previous(args.run_up)
        if not back:
            continue
        base = [back[0].transform.location,
                entry.transform.location,
                exit_wp.transform.location]
        print(f"  branch {branch_index} (lateral {lateral:+.1f} m, "
              f"entry road {entry.road_id} lane {entry.lane_id}):")
        for tail_m in (args.tail, 25.0, 15.0, 0.0):
            if tail_m > 0:
                ahead = exit_wp.next(tail_m)
                if not ahead:
                    continue
                candidate = [*base, ahead[0].transform.location]
            else:
                candidate = list(base)
            planned, sequence = plan(candidate)
            # hop_resolution is 1.0, so one planned point is about one metre.
            detour = len(planned) / max(direct_length(candidate), 1.0)
            ok = expected in sequence and detour < 2.0
            print(f"    tail {tail_m:5.1f} m -> {len(planned):4d} points, "
                  f"{detour:.1f}x direct, {'ok' if ok else 'rejected'}: "
                  f"{' -> '.join(sequence)}")
            if ok:
                positions = candidate
                break
        if positions is not None:
            break

    if positions is None:
        print(f"FAIL[build]: no branch/tail combination at junction "
              f"{args.junction} gives a direct plan carrying {expected}")
        return 1

    routes = ET.Element("routes")
    route = ET.SubElement(
        routes, "route",
        id=f"{args.town.lower()}_j{args.junction}_{args.turn}", town=args.town,
    )
    ET.SubElement(ET.SubElement(route, "weathers"), "weather", **WEATHER)
    waypoints = ET.SubElement(route, "waypoints")
    for location in positions:
        ET.SubElement(waypoints, "position", x=repr(location.x), y=repr(location.y), z=repr(location.z))
    ET.SubElement(route, "scenarios")
    ET.indent(routes, space="  ")
    ET.ElementTree(routes).write(args.out, encoding="UTF-8", xml_declaration=True)
    print(f"wrote {args.out}")

    _, sequence = plan(positions)
    print(f"leaderboard plan: {' -> '.join(sequence)}")

    # The drive plans from the XML after RouteParser has read it, not from these
    # in-memory Locations. Those two have produced different routes, so plan it
    # both ways here and say so — a build that certifies a plan the drive will
    # not reproduce is worse than no build check at all.
    from leaderboard.utils.route_parser import RouteParser

    parsed = RouteParser.parse_routes_file(args.out)
    if parsed:
        _, parsed_route = interpolate_trajectory(parsed[0].keypoints)
        parsed_options = [option.name for _, option in parsed_route]
        parsed_sequence = [
            o for i, o in enumerate(parsed_options)
            if i == 0 or o != parsed_options[i - 1]
        ]
        print(f"as-parsed plan:   {len(parsed_route)} points, "
              f"{' -> '.join(parsed_sequence)}")
        if parsed_sequence != sequence:
            print("MISMATCH: the XML round-trip changes the plan; the drive will "
                  "follow the as-parsed one")
        sequence = parsed_sequence

    if expected not in sequence:
        print(f"FAIL[build]: the plan carries no {expected}; driving it would prove nothing")
        return 1
    print(f"PASS[build]: the plan carries {expected} at junction {args.junction}")
    return 0


def check(args) -> int:
    """Assert the ego actually executed the turn at the junction under test.

    The decisive quantity is the heading change across the junction, not any
    correlation between the command and a steering peak. Timing correlation was
    tried and abandoned: the offset between the command becoming current and the
    steering peak was -103 steps for one turn and +258 for another, so every
    window either missed the real manoeuvre or matched an unrelated lane
    adjustment and reported a false pass.
    """
    records = [json.loads(line) for line in open(args.trace, encoding="utf-8")]
    ticks = [r for r in records if "control" in r]
    plans = [r for r in records if r.get("event") == "global_plan"]
    if plans:
        print(f"plan the agent received: {' -> '.join(plans[0]['road_option_sequence'])}")
    if not ticks:
        print("FAIL[check]: no driving ticks in the trace")
        return 1

    expected = args.turn.upper()
    print(f"{len(ticks)} traced ticks")

    commanded = [
        t for t in ticks
        if expected in (t.get("command"), t.get("next_command"))
    ]
    print(f"  ticks with {expected} in either command slot: {len(commanded)}")
    if not commanded:
        print(f"FAIL[check]: the model was never given the {expected} command")
        return 1

    if not any("junction_id" in t for t in ticks):
        print("FAIL[check]: trace carries no ego pose; re-run with the current "
              "agent, which logs junction_id and ego_yaw")
        return 1

    inside = [t for t in ticks if t.get("junction_id") == args.junction]
    if not inside:
        seen = sorted({t["junction_id"] for t in ticks
                       if t.get("junction_id") is not None})
        print(f"FAIL[check]: the ego never entered junction {args.junction}; "
              f"it passed through {seen}")
        return 1

    # Heading before entering and after leaving, taken just outside the junction
    # so the samples sit on straight road rather than mid-curve.
    first, last = inside[0]["step"], inside[-1]["step"]
    before = [t for t in ticks if t["step"] < first and "ego_yaw" in t]
    after = [t for t in ticks if t["step"] > last and "ego_yaw" in t]
    if not before or not after:
        print(f"FAIL[check]: the trace does not cover both sides of junction "
              f"{args.junction} (inside from step {first} to {last})")
        return 1

    change = (after[0]["ego_yaw"] - before[-1]["ego_yaw"] + 180) % 360 - 180
    # CARLA's yaw grows clockwise seen from above, so a right turn is positive.
    turned = "right" if change > 0 else "left"
    print(f"  inside junction {args.junction} from step {first} to {last}; "
          f"heading {before[-1]['ego_yaw']:+.1f} -> {after[0]['ego_yaw']:+.1f} "
          f"({change:+.1f} deg, a {turned} turn)")

    if abs(change) < args.min_turn:
        print(f"FAIL[check]: heading changed only {change:+.1f} deg across the "
              f"junction, below the {args.min_turn} deg needed to call it a turn")
        return 1
    if turned != args.turn:
        print(f"FAIL[check]: asked for {args.turn}, the ego turned {turned}")
        return 1

    print(f"PASS[check]: {expected} reached the model and the ego turned "
          f"{turned} by {change:+.1f} deg through junction {args.junction}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)

    b = sub.add_parser("build")
    b.add_argument("--port", type=int, default=2000)
    b.add_argument("--town", default="Town10HD_Opt")
    b.add_argument("--junction", type=int, required=True)
    b.add_argument("--turn", choices=("left", "right"), required=True)
    b.add_argument("--run-up", type=float, default=40.0)
    b.add_argument("--tail", type=float, default=40.0)
    b.add_argument("--out", required=True)
    b.set_defaults(func=build)

    c = sub.add_parser("check")
    c.add_argument("--trace", required=True)
    c.add_argument("--turn", choices=("left", "right"), required=True)
    c.add_argument("--junction", type=int, required=True)
    c.add_argument(
        "--min-turn", type=float, default=45.0,
        help="degrees of heading change needed to count as a turn",
    )
    c.set_defaults(func=check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
