"""scenario_orchestration/policy.py — TransFuser v6 as an `ego_policy_v1` policy.

The contract the method repository loads (`scenario_orchestration/SPEC.md`):

    build_policy(request: dict) -> policy       # request is policy.json, verbatim
    policy.act(observation: dict) -> action     # action carries a `control` block

plus the optional hooks the port uses when present: `sensors()`, `load()`,
`reset()`, `close()`, `metadata()`.

What this file is and is not
----------------------------
It is a translation layer. It owns no network, no controller and no tuning
constant: the model is built by `lead`'s own `PolicyRunner`, and the controls
are produced by `lead`'s own `WaypointTracker` / `PathSpeedTracker` through the
same modality selection `TransfuserAgent` applies. That is deliberate and it is
the same rule `carla_port/ego_driver.py` states for the port side — a policy's
published closed-loop behaviour is only reproduced if the controller that turns
its plans into pedals is the one it was published with. A steering law written
here would change the numbers while still calling them TransFuser v6's.

What it does own is the mapping from the port's observation to the model's
batch: three cameras stitched into one strip, a LiDAR sweep rasterized into the
BEV density grid by `lead`'s own `rasterize_lidar_bev`, the route reduced to the
three target points the planner is conditioned on, and the ego speed.

Radar
-----
The rig serves cameras, LiDAR and the four radars `lead`'s own
`SensorRigConfig.radars` describes. It used to serve no radar and pass a zero
tensor, on the reasoning that "the radar branch then contributes nothing rather
than something wrong". That reasoning does not hold: `_preprocess_radar_input`
zero-PADS short sweeps, so zero rows are the in-distribution encoding of "no
detection in this slot", and an all-zero block is a well-formed sweep reporting
a clear road in every direction. `RadarDetector` tokenizes it into the planning
decoder's cross-attention keys either way, and nothing downstream can tell a
clear road from an absent sensor.

Detections come back from the port in the ego frame (x forward, metres), which
is the frame `filter_and_pad_radars` bounds-checks and `_tokenize_radar` samples
BEV features in. `TFV6_RADAR=zeros` restores the old behaviour for an A/B.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

#: The rig this model is trained behind, taken from `lead`'s own
#: `SensorRigConfig`: three 384x384 pinhole cameras stitched left-to-right into
#: the 1152x384 strip `policy.transfuser.final_image_*` describes, plus the
#: LiDAR its BEV branch reads. Plain dicts, so the port does not have to be
#: importable from here.
#: `lead.config.expert.sensor_rig.SensorRigConfig.cameras`, entries 0-2 — the
#: three `TransfuserCameraConfig.input_cameras` ([PCAM_L0, PCAM_F0, PCAM_R0]) in
#: their stitch order. Copied value for value rather than approximated, because
#: every one of them was wrong before and the errors compounded:
#:
#:   was   x=-1.5  z=2.00  fov=90  yaw=-60/0/+60
#:   is    per-camera pos at z=2.25, fov=60, yaw=-57.5/0/+57.5
#:
#: `x=-1.5` is 1.5 m BEHIND the vehicle origin, which on the 4.89 m leaderboard
#: hero puts the lens inside the passenger compartment: the captured frame is
#: 40% dashboard, steering wheel and seats. The model then targets ~0 m/s on an
#: empty road, exactly as simlingo did when fed its own bonnet. Unlike simlingo,
#: `lead` has no crop to compensate — its rig simply mounts the cameras where
#: they can see out.
SENSORS: List[Dict[str, Any]] = [
    {"name": "PCAM_L0", "width": 384, "height": 384, "fov": 60.0, "yaw": -57.5,
     "x": 0.0, "y": -0.3, "z": 2.25},
    {"name": "PCAM_F0", "width": 384, "height": 384, "fov": 60.0, "yaw": 0.0,
     "x": 0.25, "y": 0.0, "z": 2.25},
    {"name": "PCAM_R0", "width": 384, "height": 384, "fov": 60.0, "yaw": 57.5,
     "x": 0.0, "y": 0.3, "z": 2.25},
    # `SensorRigConfig.lidar_pos_1` — the rig mounts at z=2.5, comfortably above
    # the hero's 1.49 m roof. At the old z=1.85 a full sweep grazed the ego's own
    # bodywork; the previous 20 Hz wedge hid that by never facing forward.
    {"name": "lidar", "kind": "sensor.lidar.ray_cast", "channels": 64,
     "range_m": 100.0, "rotation_frequency": 20.0,
     "x": 1.0, "y": 0.0, "z": 2.5},
]

#: The radar rig, copied from `lead.config.expert.sensor_rig.SensorRigConfig`
#: `radars` — mounting pose and FOV per sensor, in list order, because
#: `_preprocess_radar_input` identifies a sensor by its INDEX (`radar{i}`) and
#: writes that index into the fifth column. Reordering these silently relabels
#: every detection.
#:
#: `points_per_second` is this port's, not upstream's: the sweep only has to
#: fill `num_radar_points_per_sensor` (75) slots per sensor per 20 Hz decision,
#: and 1500/s gives that with headroom.
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

#: Set TFV6_RADAR=zeros to restore the old all-zero radar tensor. The rig is
#: new and there is no reference implementation in `lead` to check the
#: sensor->ego transform against (the training radar arrives pre-built from the
#: collection pipeline, which is not vendored here), so the previous behaviour
#: stays one environment variable away for an A/B.
RADAR_MODE = os.environ.get("TFV6_RADAR", "sensor").strip().lower()

#: Left-to-right stitch order — `AbstractPolicy.input_cameras` documents that
#: the order is the strip order, and the model is trained on exactly this one.
CAMERA_ORDER = ("PCAM_L0", "PCAM_F0", "PCAM_R0")

DEFAULT_CHECKPOINT = os.path.join(
    os.environ.get("AV_CKPT", ""), "tfv6", "resnet34_v1.5.0", "seed0")


class TransfuserV6Policy:
    """TransFuser v6, behind the `ego_policy_v1` interface."""

    def __init__(self, request: Optional[Dict[str, Any]] = None):
        self.request = dict(request or {})
        params = dict(self.request.get("parameters") or {})
        self.checkpoint = str(self.request.get("checkpoint")
                              or params.get("checkpoint")
                              or DEFAULT_CHECKPOINT)
        self.town = str(params.get("town", "Town05"))
        self.device_name = str(params.get("device", "cuda:0"))
        self.runner = None
        self.controller = None
        self.lead_config = None
        self._torch = None
        self._steps = 0
        self._notes: List[str] = []

    # ------------------------------------------------------------------ #
    # The rig
    # ------------------------------------------------------------------ #
    def sensors(self) -> List[Dict[str, Any]]:
        """What the port should attach to the ego. See `carla_sensors.py`."""
        rig = [dict(s) for s in SENSORS]
        if RADAR_MODE != "zeros":
            rig += [dict(s) for s in RADARS]
        return rig

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Build the model. Called during setup so a bad checkpoint fails
        before the simulation starts, not on tick one."""
        if self.runner is not None:
            return
        import torch
        import yaml
        from lead.config import load_lead_config
        from lead.evaluation.inference.policy_runner import PolicyRunner
        from lead.evaluation.inference.trackers import (PathSpeedTracker,
                                                        WaypointTracker)

        self._torch = torch
        config_path = os.path.join(self.checkpoint, "config.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"no config.yaml in checkpoint dir {self.checkpoint!r}; "
                "point `checkpoint` at a directory holding config.yaml and "
                "one model_*.pth")
        with open(config_path, encoding="utf-8") as handle:
            stored = yaml.safe_load(handle)
        # How it trained, not how we evaluate — the same drop
        # lead/api/abstract_driving_agent.py::setup makes.
        stored.pop("evaluation", None)
        self.lead_config = load_lead_config(loaded_config=stored,
                                            raise_on_unknown_key=False)
        device = torch.device(self.device_name)
        self.runner = PolicyRunner(lead_config=self.lead_config,
                                   model_path=self.checkpoint, device=device)
        self.controller = _Controller(self.lead_config,
                                      WaypointTracker(self.lead_config),
                                      PathSpeedTracker(self.lead_config))

    def reset(self) -> None:
        self._steps = 0

    def close(self) -> None:
        self.runner = None
        self.controller = None

    # ------------------------------------------------------------------ #
    # One decision
    # ------------------------------------------------------------------ #
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.runner is None:
            self.load()
        torch = self._torch
        batch = self._batch(observation)
        prediction = self.runner.forward(batch)
        steer, throttle, brake = self.controller.control(prediction, batch)
        self._steps += 1

        waypoints = prediction.future_waypoints
        return {
            "control": {"throttle": float(throttle), "steer": float(steer),
                        "brake": float(brake)},
            "waypoints": (waypoints[0].float().cpu().numpy().tolist()
                          if waypoints is not None else None),
            "target_speed_mps": (
                float(prediction.target_speed_scalar[0])
                if prediction.target_speed_scalar is not None else None),
            "meta": {"policy": "tfv6", "step": self._steps,
                     "radar": ("zeros (TFV6_RADAR=zeros)" if RADAR_MODE == "zeros"
                               else f"{len(RADARS)} x sensor.other.radar, "
                                    "ego frame, lead's own preprocessing")},
        }

    # ------------------------------------------------------------------ #
    # observation -> model batch
    # ------------------------------------------------------------------ #
    def _batch(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        torch = self._torch
        device = self.runner.device
        cfg = self.lead_config.policy.transfuser
        cameras = dict((observation.get("sensor") or {}).get("cameras") or {})

        rgb = self._stitch(cameras, cfg)
        lidar = self._lidar_bev(cameras, cfg)
        n_radar = (cfg.num_radar_points_per_sensor
                   * self.lead_config.expert.sensor_rig.num_radar_sensors)

        previous, target, following = _target_points(observation)
        speed = float(((observation.get("ego") or {}).get("speed_mps")) or 0.0)

        def tensor(array):
            return torch.as_tensor(np.asarray(array), dtype=torch.float32,
                                   device=device)[None]

        return {
            "town": [self.town],
            "rgb": tensor(rgb),
            "rasterized_lidar": tensor(lidar),
            "radar": tensor(self._radar(cameras, cfg, n_radar)),
            "previous_target_point": tensor(previous),
            "target_point": tensor(target),
            "next_target_point": tensor(following),
            "speed": torch.tensor([speed], dtype=torch.float32, device=device),
        }

    def _stitch(self, cameras: Dict[str, Any], cfg) -> np.ndarray:
        """The three views as one CHW strip, in the rig's own left-to-right
        order. A camera the port could not attach becomes black rather than
        shifting every other view into the wrong third of the strip."""
        height, width = cfg.final_image_height, cfg.final_image_width
        per = width // len(CAMERA_ORDER)
        views = []
        for name in CAMERA_ORDER:
            image = cameras.get(name)
            if image is None:
                self._note(f"camera {name} missing; substituted black")
                image = np.zeros((height, per, 3), dtype=np.uint8)
            views.append(np.asarray(image)[:height, :per, :3])
        strip = np.concatenate(views, axis=1)          # H x W x 3
        return np.transpose(strip, (2, 0, 1)).astype(np.float32)

    def _radar(self, sensors: Dict[str, Any], cfg, n_radar: int) -> np.ndarray:
        """The `(300, 5)` radar block, built by `lead`'s own preprocessing.

        `_preprocess_radar_input` filters each sensor's detections to the BEV
        bounds, truncates or ZERO-PADS to `num_radar_points_per_sensor`, and
        appends the sensor index as a fifth column. Zero rows are therefore the
        in-distribution "no detection in this slot" value — which is exactly why
        the old all-zero tensor was not a safe stand-in for a missing rig: it is
        a well-formed sweep reporting a clear road in every direction, and the
        planner has no way to read it as "no radar fitted".

        A sensor that failed to attach still contributes its zero block, so the
        sensor-index column stays aligned with `sensor_rig.radars`.
        """
        if RADAR_MODE == "zeros":
            self._note("radar disabled (TFV6_RADAR=zeros): the tensor is all "
                       "zeros, which the planner reads as a clear road")
            return np.zeros((n_radar, 5), dtype=np.float32)
        from lead.policy.transfuser.dataloader.features import (
            _preprocess_radar_input)
        by_sensor, missing = {}, []
        for index, spec in enumerate(RADARS, start=1):
            points = sensors.get(spec["name"])
            if points is None:
                missing.append(spec["name"])
                points = np.zeros((0, 4), dtype=np.float16)
            # float16, not float32: `_preprocess_radar_input` is jaxtyping-
            # annotated `dict[str, Float16[ndarray, "_ 4"]]` and the runtime
            # checker rejects anything else outright.
            by_sensor[f"radar{index}"] = np.asarray(points, dtype=np.float16)
        if missing:
            self._note(f"radar sweep missing for {', '.join(missing)}; those "
                       "sensors contribute zero rows")
        blocks = _preprocess_radar_input(self.lead_config, by_sensor)
        if not blocks:
            return np.zeros((n_radar, 5), dtype=np.float32)
        return np.concatenate(blocks, axis=0).astype(np.float32)

    def _lidar_bev(self, sensors: Dict[str, Any], cfg) -> np.ndarray:
        """The BEV density grid, rasterized by `lead`'s own function."""
        points = sensors.get("lidar")
        if points is None:
            self._note("no LiDAR sweep in the observation; BEV grid is zeros")
            return np.zeros((1, cfg.lidar_height_pixel, cfg.lidar_width_pixel),
                            dtype=np.float32)
        from lead.policy.transfuser.dataloader.features import rasterize_lidar_bev
        grid = rasterize_lidar_bev(
            self.lead_config, np.asarray(points)[:, :3],
            remove_ground_plane=cfg.remove_lidar_ground_points)
        return np.asarray(grid, dtype=np.float32)[None]

    def _note(self, text: str) -> None:
        if text not in self._notes:
            self._notes.append(text)

    # ------------------------------------------------------------------ #
    def metadata(self) -> Dict[str, Any]:
        return {
            "policy": "tfv6",
            "upstream": "kesai-labs/lead (TransFuser v6)",
            "checkpoint": self.checkpoint,
            "observation_space": "sensor",
            "action_space": "control",
            "controller": "lead WaypointTracker + PathSpeedTracker "
                          "(the model's own, unmodified)",
            "decisions": self._steps,
            "notes": list(self._notes),
        }


class _Controller:
    """`TransfuserAgent`'s control selection, without a CARLA agent around it.

    Subclasses the real agent to inherit `_build_agent_prediction` and
    `_post_process_target_speed` unmodified — the same trick, and for the same
    reason, as `carla_port/carla_ego.EgoPolicy` borrowing drivev2's law.
    `AbstractDrivingAgent.__init__` is never called: it would need a CARLA
    world, a route and a leaderboard around it.
    """

    def __init__(self, lead_config, waypoint_tracker, path_speed_tracker):
        from lead.evaluation.agents.transfuser.transfuser_agent import TransfuserAgent

        class _Bare(TransfuserAgent):
            def __init__(self_inner):        # noqa: N805 - deliberately bare
                pass

        self._agent = _Bare()
        self._agent.lead_config = lead_config
        self._agent.waypoint_tracker = waypoint_tracker
        self._agent.path_speed_tracker = path_speed_tracker

    def control(self, prediction, batch):
        agent_prediction = self._agent._build_agent_prediction(prediction, batch)
        return (float(agent_prediction.steer), float(agent_prediction.throttle),
                float(agent_prediction.brake))


def _target_points(observation: Dict[str, Any]):
    """The three route points the planner is conditioned on.

    The port's observation carries a dense route in the ego frame
    (`carla_obs.py`: one point per metre from 2.5 m). TransFuser's planner wants
    a sparse target point rather than the whole polyline, so the route is
    sampled at roughly the spacing the leaderboard's route planner produces.
    """
    route = observation.get("route") or []
    points = [(float(p[0]), float(p[1])) for p in route
              if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not points:
        return (0.0, 0.0), (8.0, 0.0), (16.0, 0.0)
    def at(index):
        return points[min(index, len(points) - 1)]
    return at(0), at(7), at(15)


def build_policy(request: Dict[str, Any]) -> TransfuserV6Policy:
    """The factory `ego_policy_v1` requires. `request` is policy.json verbatim."""
    return TransfuserV6Policy(request)
