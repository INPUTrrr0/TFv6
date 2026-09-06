"""The closed-loop agent this test drives, with three additions to upstream's.

`lead.inference.sensor_agent.SensorAgent` is used as-is for the driving; what is
added here is what the test needs to make a claim about it.

1. **The GNSS projection.** CARLA moved its GNSS sensor from an equatorial to a
   transverse Mercator in 0.9.16. This branch targets 0.9.15 and implements only
   the equatorial inverse, so on a 0.9.16 server every ego position is decoded
   with the wrong projection. The correction is applied only when the server
   says it is needed, and only to the GNSS *sensor* decode in `base_agent` --
   `route_planner` decodes the leaderboard's GPS-encoded plan, which the
   leaderboard writes with an equatorial projection whatever CARLA does.

2. **An empty road, on request.** The leaderboard adds `BackgroundBehavior`
   unconditionally (`route_scenario.py`), and there is no flag for a clear
   world. `TFV6_NO_TRAFFIC=1` removes the other vehicles so a manoeuvre can be
   demonstrated in isolation. It is not how you score a route.

3. **A trace.** One JSONL record per command change and every `NAV_PROBE_EVERY`
   ticks: the discrete command the network is conditioned on, the three target
   points, the control, and the ego's pose and junction. `junction_turn_test.py
   check` reads it.
"""

from __future__ import annotations

import json
import math
import os
import re
import types

import carla
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import base_agent as _base_agent
from lead.common import common_utils as _common_utils
from lead.inference.sensor_agent import SensorAgent

LOG_PATH = os.environ.get("NAV_PROBE_LOG", "nav_probe.jsonl")
EVERY_N_TICKS = int(os.environ.get("NAV_PROBE_EVERY", "20"))
NO_TRAFFIC = os.environ.get("TFV6_NO_TRAFFIC", "0") == "1"

#: command_to_one_hot's index order: the RoadOption enum value minus one.
COMMAND_NAMES = ["LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW",
                 "CHANGELANELEFT", "CHANGELANERIGHT"]

#: Which CARLA versions use which GNSS projection. A version outside this table
#: is not guessed at -- a source build reporting a git hash, for instance, must
#: say which it is through TFV6_GNSS_TRANSVERSE_MERCATOR.
_PROJECTION_BY_VERSION = {"0.9.15": False, "0.9.16": True}


def get_entry_point():  # the leaderboard's agent hook
    return "JunctionTurnAgent"


def _convert_tmerc_gnss_to_carla(gnss, lat_ref: float, lon_ref: float):
    """Inverse of CARLA 0.9.16's GNSS sensor: spherical transverse Mercator."""
    earth_radius_equa = 6378137.0
    k0 = 0.9996
    lat, lon, _ = gnss
    phi = math.radians(lat)
    delta_lambda = math.radians(lon - lon_ref)
    b = math.cos(phi) * math.sin(delta_lambda)
    x = 0.5 * k0 * earth_radius_equa * math.log((1 + b) / (1 - b))
    y = (k0 * earth_radius_equa
         * (math.atan(math.tan(phi) / math.cos(delta_lambda))
            - math.radians(lat_ref)))
    return np.array([x, y, gnss[2]])


def _uses_transverse_mercator(client) -> bool:
    """Whether this server's GNSS needs the 0.9.16 inverse."""
    version = client.get_server_version()
    match = re.match(r"\d+\.\d+\.\d+", version)
    key = match.group() if match else version
    if key in _PROJECTION_BY_VERSION:
        return _PROJECTION_BY_VERSION[key]
    override = os.environ.get("TFV6_GNSS_TRANSVERSE_MERCATOR")
    if override is not None:
        return override == "1"
    raise RuntimeError(
        f"CARLA server reports version {version!r}, which is neither 0.9.15 nor "
        f"0.9.16, so its GNSS projection is unknown. Decoding with the wrong one "
        f"mislocalizes the ego silently. Set TFV6_GNSS_TRANSVERSE_MERCATOR=1 or "
        f"=0 -- measure it if you are unsure, by inverting a stationary GNSS "
        f"reading both ways and keeping the one that returns the true pose.")


class JunctionTurnAgent(SensorAgent):
    """Upstream's SensorAgent, instrumented and version-corrected."""

    def setup(self, path_to_conf_file, *args, **kwargs):
        result = super().setup(path_to_conf_file, *args, **kwargs)
        if _uses_transverse_mercator(CarlaDataProvider.get_client()):
            # Swap only base_agent's view of common_utils, so route_planner
            # keeps the equatorial inverse its input is encoded with.
            shim = types.ModuleType("common_utils_tmerc")
            shim.__dict__.update(_common_utils.__dict__)
            shim.convert_gps_to_carla = _convert_tmerc_gnss_to_carla
            _base_agent.common_utils = shim
            print("[agent] CARLA >= 0.9.16: GNSS decoded with a transverse "
                  "Mercator")
        return result

    def _emit(self, record: dict) -> None:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Record the plan that arrives, not the one we expected."""
        options = [o.name for _, o in global_plan_world_coord if o is not None]
        compressed = [o for i, o in enumerate(options)
                      if i == 0 or o != options[i - 1]]
        self._emit({"event": "global_plan",
                    "n_dense": len(global_plan_world_coord),
                    "road_option_sequence": compressed})
        required = os.environ.get("TFV6_REQUIRE_COMMAND")
        if required and required.upper() not in compressed:
            raise RuntimeError(
                f"the plan handed to the agent carries no {required.upper()}; "
                f"it is {' -> '.join(compressed)}. Refusing to drive a route "
                f"that cannot exercise the manoeuvre under test.")
        return super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def tick(self, input_data: dict) -> dict:
        """Snapshot the navigation conditioning from what `tick` returns.

        Not from inside `set_target_points`: that runs twice on a tick whose
        target points bunch, and `target_point_next` is overwritten after it.
        """
        data = super().tick(input_data)

        def name(one_hot):
            index = int(np.argmax(np.asarray(one_hot)))
            return COMMAND_NAMES[index] if index < len(COMMAND_NAMES) else str(index)

        def point(key):
            value = data.get(key)
            return None if value is None else np.asarray(value).reshape(-1).tolist()

        self._nav = {"command": name(data["command"]),
                     "next_command": name(data["next_command"]),
                     "target_point_previous": point("target_point_previous"),
                     "target_point": point("target_point"),
                     "target_point_next": point("target_point_next"),
                     "speed_mps": float(np.asarray(data["speed"]).reshape(-1)[0])}
        return data

    def run_step(self, input_data: dict, _, __=None) -> carla.VehicleControl:
        control = super().run_step(input_data, _, __)
        if NO_TRAFFIC and self.step % 10 == 0:
            self._clear_traffic()
        nav = getattr(self, "_nav", None)
        # Always record a command change: a junction manoeuvre can be shorter
        # than the sampling interval, and a command never recorded looks exactly
        # like a command never issued.
        changed = nav is not None and nav != getattr(self, "_last_nav", None)
        if nav is not None:
            self._last_nav = dict(nav)
        if nav is not None and (changed or self.step % EVERY_N_TICKS == 0):
            self._emit({"step": int(self.step), **nav,
                        "control": {"throttle": control.throttle,
                                    "steer": control.steer,
                                    "brake": control.brake},
                        **self._ego_pose()})
        return control

    def _clear_traffic(self) -> None:
        world = getattr(self, "_world", None)
        ego = getattr(self, "_vehicle", None)
        if world is None or ego is None:
            return
        try:
            for actor in world.get_actors().filter("vehicle.*"):
                if actor.id != ego.id:
                    actor.destroy()
            for walker in world.get_actors().filter("walker.*"):
                walker.destroy()
        except RuntimeError:
            pass  # an actor already gone is the expected race, not an error

    def _ego_pose(self) -> dict:
        vehicle = getattr(self, "_vehicle", None)
        world = getattr(self, "_world", None)
        if vehicle is None or world is None:
            return {}
        try:
            transform = vehicle.get_transform()
            waypoint = world.get_map().get_waypoint(transform.location)
            junction = waypoint.get_junction() if waypoint.is_junction else None
            return {"ego_yaw": round(transform.rotation.yaw, 2),
                    "junction_id": junction.id if junction is not None else None}
        except RuntimeError:
            return {}
