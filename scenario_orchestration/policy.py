"""This checkpoint behind the `ego_policy_v1` interface, command and all.

A harness that owns the CARLA world can drive this policy without the CARLA
Leaderboard: `build_policy(request)` returns an object whose `sensors()`
declares the rig to attach and whose `act(observation)` returns a control.

The interesting part is the navigation command. This branch's network requires
one — `use_discrete_command` gates both `command_encoder` and the size of
`status_pos_embedding`, so the released weights will not load without it — but a
command is never set directly, not even under the Leaderboard: CARLA's
`GlobalRoutePlanner` derives it from route geometry and the leaderboard carries
it forward.

So it is derived here the same way. A harness supplies `observation["route"]`,
the ego's intended path in the ego frame; its heading change names the
manoeuvre, and that fills the same two input slots the leaderboard would have.
A harness that sends no route gets `LANEFOLLOW`, which is the same default
`command_to_one_hot` gives an absent command.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# The rig, copied from this repository's own SensorRigConfig. Camera order is
# the stitch order the model was trained on, and radar order is load-bearing:
# preprocess_radar_input writes each sensor's INDEX into the fifth column, so
# reordering these silently relabels every detection.
SENSORS: List[Dict[str, Any]] = [
    {"name": "PCAM_L0", "width": 384, "height": 384, "fov": 60.0, "yaw": -57.5,
     "x": 0.0, "y": -0.3, "z": 2.25},
    {"name": "PCAM_F0", "width": 384, "height": 384, "fov": 60.0, "yaw": 0.0,
     "x": 0.25, "y": 0.0, "z": 2.25},
    {"name": "PCAM_R0", "width": 384, "height": 384, "fov": 60.0, "yaw": 57.5,
     "x": 0.0, "y": 0.3, "z": 2.25},
    {"name": "lidar", "kind": "sensor.lidar.ray_cast", "channels": 64,
     "range_m": 100.0, "rotation_frequency": 20.0,
     "x": 1.0, "y": 0.0, "z": 2.5},
]
RADARS: List[Dict[str, Any]] = [
    {"name": "radar1", "kind": "sensor.other.radar", "x": 2.6, "z": 0.60,
     "yaw": -45.0, "horizontal_fov": 90.0, "vertical_fov": 0.1,
     "range_m": 100.0, "points_per_second": 1500},
    {"name": "radar2", "kind": "sensor.other.radar", "x": 2.6, "z": 0.60,
     "yaw": 45.0, "horizontal_fov": 90.0, "vertical_fov": 0.1,
     "range_m": 100.0, "points_per_second": 1500},
    {"name": "radar3", "kind": "sensor.other.radar", "x": -2.6, "z": 0.60,
     "yaw": 135.0, "horizontal_fov": 90.0, "vertical_fov": 0.1,
     "range_m": 100.0, "points_per_second": 1500},
    {"name": "radar4", "kind": "sensor.other.radar", "x": -2.6, "z": 0.60,
     "yaw": 225.0, "horizontal_fov": 90.0, "vertical_fov": 0.1,
     "range_m": 100.0, "points_per_second": 1500},
]
CAMERA_ORDER = ("PCAM_L0", "PCAM_F0", "PCAM_R0")

def _default_checkpoint() -> str:
    """$TFV6_CHECKPOINT, else the first cvpr2026 weights directory that exists.

    The in-repo location `outputs/checkpoints/tfv6_resnet34` is where a
    training run writes, and on this cluster the released weights were never
    placed there -- they live in the shared store under
    `$AV_CKPT/tfv6_cvpr2026/tfv6_resnet34`. Returning a path that does not
    exist turned every caller that passed no `checkpoint` into a crash, so the
    shared store is tried too, and the in-repo path stays last so a local
    training output still wins when there is one.
    """
    env = os.environ.get("TFV6_CHECKPOINT")
    if env:
        return env
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [os.path.join(repo, "outputs", "checkpoints", "tfv6_resnet34")]
    if os.environ.get("AV_CKPT"):
        candidates.append(os.path.join(os.environ["AV_CKPT"], "tfv6_cvpr2026",
                                       "tfv6_resnet34"))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    return candidates[-1]


DEFAULT_CHECKPOINT = _default_checkpoint()

# command_to_one_hot's index order: the RoadOption enum value minus one.
# RUNBOOK.md section 1 has the table.
LEFT, RIGHT, STRAIGHT, LANEFOLLOW = 0, 1, 2, 3
COMMAND_DIM = 6

#: Degrees of heading change across the looked-at stretch of route below which
#: the manoeuvre is not a turn. A junction turn runs to 70-90 deg (measured at
#: junction 189: -74.3 and +89.0), and lane-keeping wander stays well under 20.
TURN_DEGREES = 25.0


def _one_hot(index: int) -> np.ndarray:
    vector = np.zeros(COMMAND_DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _heading_change(points: List[Tuple[float, float]]) -> float:
    """Signed heading change along a polyline, in degrees.

    The route is in the ego frame, which is CARLA's: x forward, **y right**, so
    a positive change turns right. That is the same convention
    junction_turn_test.py classifies branches with.
    """
    if len(points) < 3:
        return 0.0
    def bearing(a, b):
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    start = bearing(points[0], points[min(2, len(points) - 1)])
    end = bearing(points[-3], points[-1])
    return (end - start + 180.0) % 360.0 - 180.0


def _classify(change: float) -> int:
    if abs(change) < TURN_DEGREES:
        return LANEFOLLOW
    return RIGHT if change > 0 else LEFT


class TransfuserV6CvprPolicy:
    """TFv6 from the CVPR branch, behind `ego_policy_v1`."""

    #: Consecutive ticks a changed command must hold before it is issued.
    COMMAND_HOLD_TICKS = 5

    def __init__(self, request: Optional[Dict[str, Any]] = None):
        self.request = dict(request or {})
        params = dict(self.request.get("parameters") or {})
        self.checkpoint = str(self.request.get("checkpoint")
                              or params.get("checkpoint")
                              or DEFAULT_CHECKPOINT)
        self.town = str(params.get("town", "Town10HD"))
        self.device_name = str(params.get("device", "cuda:0"))
        self.inference = None
        self.training_config = None
        self.config_expert = None
        self._torch = None
        self._steps = 0
        self._commands: List[str] = []
        self._last_nav = None
        self._command = LANEFOLLOW
        self._pending = 0

    def sensors(self) -> List[Dict[str, Any]]:
        return [dict(s) for s in SENSORS] + [dict(s) for s in RADARS]

    def load(self) -> None:
        """Build the model, so a bad checkpoint fails before the simulation."""
        if self.inference is not None:
            return
        import torch
        from lead.expert.config_expert import ExpertConfig
        from lead.inference.closed_loop_inference import ClosedLoopInference
        from lead.inference.config_closed_loop import ClosedLoopConfig
        from lead.training.config_training import TrainingConfig

        self._torch = torch
        config_path = os.path.join(self.checkpoint, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"no config.json in checkpoint dir {self.checkpoint!r}; the "
                "cvpr2026 branch stores its training config as JSON, not the "
                "config.yaml the main stack writes")
        with open(config_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        self.training_config = TrainingConfig(stored)
        if not getattr(self.training_config, "use_discrete_command", False):
            raise RuntimeError(
                f"checkpoint {self.checkpoint!r} was trained without a discrete "
                "command; this adapter exists to supply one, so it is the wrong "
                "checkpoint for it")
        self.config_expert = ExpertConfig()
        self.inference = ClosedLoopInference(
            config_training=self.training_config,
            config_closed_loop=ClosedLoopConfig(),
            config_expert=self.config_expert,
            model_path=self.checkpoint,
            device=torch.device(self.device_name),
            prefix="model",
        )

    def reset(self) -> None:
        self._steps = 0
        self._commands = []
        self._last_nav = None
        self._command = LANEFOLLOW
        self._pending = 0

    def close(self) -> None:
        self.inference = None

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.inference is None:
            self.load()
        batch, commands = self._batch(observation)
        prediction = self.inference.forward(data=batch)
        self._steps += 1
        if not self._commands or self._commands[-1] != commands[0]:
            self._commands.append(commands[0])
        # The three target points go into meta so a harness can show what the
        # network was actually conditioned on. A video that shows only the world
        # cannot distinguish "drove straight through a junction it was told to
        # turn at" from "was never told to turn".
        points = [batch[key][0].tolist() for key in
                  ("target_point_previous", "target_point", "target_point_next")]
        return {
            "control": {"throttle": float(prediction.throttle),
                        "steer": float(prediction.steer),
                        "brake": float(prediction.brake)},
            "meta": {"policy": "tfv6", "step": self._steps,
                     "command": commands[0], "next_command": commands[1],
                     "target_points": points},
        }

    # ---------------------------------------------------------------- #
    def _batch(self, observation: Dict[str, Any]):
        torch = self._torch
        device = self.inference.device
        # The port files every attached sensor under `cameras`, lidar and radar
        # included; the key is the name declared in sensors().
        sensors = dict((observation.get("sensor") or {}).get("cameras") or {})

        previous, target, following, commands = self._navigation(observation)
        speed = float(((observation.get("ego") or {}).get("speed_mps")) or 0.0)

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)

        batch = {
            "rgb": tensor(self._stitch(sensors))[None],
            "rasterized_lidar": tensor(self._lidar(sensors)),
            "target_point_previous": tensor(previous).view(1, 2),
            "target_point": tensor(target).view(1, 2),
            "target_point_next": tensor(following).view(1, 2),
            "speed": tensor([speed]).view(1),
            "command": tensor(_one_hot(commands[2])).view(1, COMMAND_DIM),
            "next_command": tensor(_one_hot(commands[3])).view(1, COMMAND_DIM),
            "town": np.array([self.town]),
        }
        if getattr(self.training_config, "use_radars", False):
            batch["radar"] = tensor(self._radar(sensors))[None]
        return batch, commands

    def _navigation(self, observation: Dict[str, Any]):
        """Three target points and two commands, both from the port's route.

        `command` describes the manoeuvre the ego is in and `next_command` the
        one after, so the near half of the route decides the first and the whole
        of it the second. While the turn is still ahead only `next_command`
        names it; once the ego is in the bend both do — the same ordering the
        leaderboard produces, where next_command announces a turn well before
        command does.
        """
        route = observation.get("route") or []
        points = [(float(p[0]), float(p[1])) for p in route
                  if isinstance(p, (list, tuple)) and len(p) >= 2]
        names = ("LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW")

        if not points:
            # A route that has run out is not a route that says "go straight".
            # Padding a target point 16 m ahead of the ego, as this first did,
            # is an instruction, and the ego obeys it: it drove straight out of
            # a left turn once the reference path ended. With nothing to aim at,
            # hold the last target points that were real and keep the last
            # command, so the manoeuvre in progress is not cancelled by the
            # route simply ending.
            if self._last_nav is not None:
                return self._last_nav
            return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (
                names[LANEFOLLOW], names[LANEFOLLOW], LANEFOLLOW, LANEFOLLOW)

        def at(index):
            return points[min(index, len(points) - 1)]

        # The manoeuvre is the whole visible route's heading change, not a
        # near-window's. A window that moves with the ego re-decides the command
        # every tick: at 20 Hz this flickered LEFT -> LANEFOLLOW -> LEFT while
        # the ego approached a junction it was supposed to turn at. Upstream's
        # command does not flicker because it is attached to a target point
        # rather than recomputed from wherever the ego happens to be.
        raw = _classify(_heading_change(points))
        current = self._stabilise(raw)
        nav = (at(0), at(7), at(15),
               (names[current], names[raw], current, raw))
        self._last_nav = nav
        return nav

    def _stabilise(self, raw: int) -> int:
        """Only accept a changed command after it has held for a few ticks.

        Hysteresis, because the classification is a threshold on a continuous
        quantity and the route wobbles as the ego moves along it. Without this a
        single noisy tick can hand the network a LANEFOLLOW in the middle of a
        turn.
        """
        if raw == self._command:
            self._pending = 0
            return self._command
        self._pending += 1
        if self._pending >= self.COMMAND_HOLD_TICKS:
            self._command = raw
            self._pending = 0
        return self._command

    def _stitch(self, sensors: Dict[str, Any]) -> np.ndarray:
        """The three views as one CHW strip, left to right in rig order."""
        config = self.training_config
        height = config.final_image_height
        width = config.final_image_width
        per = width // len(CAMERA_ORDER)
        views = []
        for name in CAMERA_ORDER:
            image = sensors.get(name)
            if image is None:
                views.append(np.zeros((height, per, 3), dtype=np.uint8))
                continue
            array = np.asarray(image)[..., :3]
            views.append(array.astype(np.uint8))
        strip = np.concatenate(views, axis=1)
        return np.transpose(strip, (2, 0, 1))

    def _lidar(self, sensors: Dict[str, Any]) -> np.ndarray:
        """The BEV grid, through cvpr2026's own rasterizer and its train-time
        compression round trip — sensor_agent.tick does both, and skipping the
        second is a train/test mismatch rather than an optimisation."""
        from lead.data_loader import training_cache
        from lead.data_loader.carla_dataset_utils import rasterize_lidar

        config = self.training_config
        points = sensors.get("lidar")
        if points is None:
            return np.zeros((1, 1, config.lidar_resolution_height,
                             config.lidar_resolution_width), dtype=np.float32)
        cloud = np.asarray(points)[:, :3].astype(np.float64)
        for axis, precision in enumerate((self.config_expert.point_precision_x,
                                          self.config_expert.point_precision_y,
                                          self.config_expert.point_precision_z)):
            cloud[:, axis] = np.round(cloud[:, axis] / precision) * precision
        grid = rasterize_lidar(config=config, lidar=cloud)[..., None]
        grid = training_cache.compress_float_image(grid, config)
        grid = training_cache.decompress_float_image(grid).squeeze()[None, None]
        return np.asarray(grid, dtype=np.float32)

    def _radar(self, sensors: Dict[str, Any]) -> np.ndarray:
        """The radar block, through cvpr2026's own preprocessing."""
        from lead.data_loader import carla_dataset_utils

        raw = {}
        for index, spec in enumerate(RADARS, start=1):
            points = sensors.get(spec["name"])
            raw[f"radar{index}"] = (
                np.zeros((0, 4), dtype=np.float32) if points is None
                else np.asarray(points, dtype=np.float32))
        blocks = carla_dataset_utils.preprocess_radar_input(
            self.training_config, raw)
        if not blocks:
            n = (self.training_config.num_radar_points_per_sensor
                 * self.training_config.num_radar_sensors)
            return np.zeros((n, 5), dtype=np.float32)
        return np.concatenate(blocks, axis=0).astype(np.float32)

    def metadata(self) -> Dict[str, Any]:
        return {"policy": "tfv6", "checkpoint": self.checkpoint,
                "commands_issued": self._commands,
                "note": "command derived from observation['route'] geometry, "
                        "the same information GlobalRoutePlanner derives it "
                        "from upstream"}


def build_policy(request: Dict[str, Any]) -> TransfuserV6CvprPolicy:
    """The factory `ego_policy_v1` requires; `request` is policy.json verbatim."""
    return TransfuserV6CvprPolicy(request)
