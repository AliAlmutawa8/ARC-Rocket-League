"""Headless Gymnasium environment for training a Rocket League car with RL.

Design notes
------------
* Units are centimetres and seconds. Field geometry matches ``simulator.py``.
* The car is planar (kinematic bicycle model with motor and steering-servo lag).
  The ball is 2.5D: pymunk handles x/y, while height ``h`` and vertical speed
  ``vh`` are integrated here, so a hit can pop the ball up and off the floor.
* Observations are egocentric (car frame) and delayed/noisy like a real
  perception-to-control loop. Rewards use the true, undelayed state.
* Everything physical is randomised per episode (see ``Randomization``).
* Approach shaping (``car_progress``, ``align``) targets a predicted ball-intercept
  point, not the ball's current position, so a moving ball is led rather than chased
  from behind (see ``_intercept_point``). This only affects the shaping reward, not the
  physics or the observation.
* Curriculum review (replaying earlier stages so skills aren't forgotten as training
  moves on) is stagewise: how often a review episode happens grows with how far training
  has progressed (``review_prob``), and which earlier stage gets replayed is weighted
  toward the more fundamental ones, "aim" (stage 1) especially, rather than picked
  uniformly (``review_weights``).
* Convention: +x is east, angles are counter-clockwise in sim coordinates and
  "right" is +90 degrees from the heading. That is right on screen because
  pygame's y axis points down. It matches ``CarAction``: positive steer = right.
* The agent is team 1 at the left and attacks the goal at x = FIELD_WIDTH.

Curriculum stages (``set_stage``)
    0 reach          stationary ball, success = touch it
    1 aim            stationary ball, success = touch it so it heads at the goal
    2 score_static   stationary ball, success = score
    3 score_rolling  rolling ball, success = score
    4 score_defender rolling ball plus a scripted defender, success = score
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields

import gymnasium as gym
import numpy as np
import pymunk
from gymnasium import spaces

# --- Field (same numbers as simulator.py, kept here to avoid importing pygame)
FIELD_WIDTH = 426.72
FIELD_HEIGHT = 304.8
GOAL_HEIGHT = 81.28
GOAL_DEPTH = 25.4
SIDE_WALL = (FIELD_HEIGHT - GOAL_HEIGHT) / 2
GOAL_Y_MIN = SIDE_WALL
GOAL_Y_MAX = SIDE_WALL + GOAL_HEIGHT
OPP_GOAL = np.array([FIELD_WIDTH, FIELD_HEIGHT / 2])
OWN_GOAL = np.array([GOAL_DEPTH, FIELD_HEIGHT / 2])

# --- 2.5D specs
WALL_HEIGHT = 10.0  # a ball higher than this clears the walls
CROSSBAR_HEIGHT = 20.0  # a ball higher than this over the line is not a goal
CAR_HEIGHT = 5.0  # a ball higher than this passes over the cars
GRAVITY = 981.0

# --- Car / ball specs
CAR_SIZE = (16.5, 8.5)
CAR_MASS = 5.0
BALL_RADIUS = 6.85 / 2

CAT_CAR, CAT_BALL, CAT_WALL = 1, 2, 4

# Observation scales (values are divided by these, then clipped to +-3)
POS_SCALE, VEL_SCALE, GOAL_SCALE = 300.0, 200.0, 400.0
CAR_VEL_SCALE, YAW_SCALE, H_SCALE, VH_SCALE = 150.0, 4.0, 10.0, 100.0

STAGE_NAMES = ("reach", "aim", "score_static", "score_rolling", "score_defender")
DEFAULT_MAX_STEPS = {0: 200, 1: 200, 2: 300, 3: 300, 4: 400}
ACTION_HISTORY = 2
OBS_DIM = 2 + 2 + 2 + 2 + 1 + 2 + 2 + 2 + 5 + 2 * ACTION_HISTORY

# Stagewise review: probability of a review episode, indexed by the *current* training
# stage (index 0 is unused - stage 0 has no earlier stage to review).
DEFAULT_REVIEW_PROBS = (0.0, 0.25, 0.35, 0.40, 0.45)
# Relative sampling weight of each stage as a review target, applied over whichever
# earlier stages are eligible (index < current stage). Higher = reviewed more often.
# "aim" (1) is weighted highest since positioning/approach is the skill every later
# stage builds on; "reach" (0) next since it's the most basic skill of all.
DEFAULT_REVIEW_WEIGHTS = (1.5, 3.0, 1.3, 1.0, 1.0)

# Reward-shaping-only ball-intercept lead (see ``_intercept_point``).
LEAD_TIME_MAX = 1.2  # s: cap on the lookahead so an uncatchable ball doesn't diverge
LEAD_SPEED_FRAC = 0.85  # assumed pursuit speed, as a fraction of the car's current top speed


@dataclass
class VehicleParams:
    """Nominal car dynamics. Calibrate these against the real car (system ID)."""
    wheelbase: float = 12.0  # cm
    max_steer_deg: float = 30.0
    v_max: float = 150.0  # cm/s
    accel: float = 300.0  # cm/s^2
    brake: float = 600.0  # cm/s^2
    servo_rate_dps: float = 400.0  # steering slew rate


@dataclass
class Randomization:
    """Per-episode (lo, hi) ranges. ``enabled=False`` uses midpoints and no noise."""
    enabled: bool = True
    motor_scale: tuple = (0.8, 1.2)  # scales accel and top speed
    servo_scale: tuple = (0.7, 1.3)  # scales steering slew rate
    ball_mass: tuple = (0.07, 0.13)
    ball_decel: tuple = (4.0, 12.0)  # rolling deceleration, cm/s^2
    ball_friction: tuple = (0.3, 0.8)
    ball_elasticity: tuple = (0.5, 0.9)
    floor_restitution: tuple = (0.3, 0.7)
    pop_coeff: tuple = (0.0, 0.4)  # vertical speed gained per unit closing speed
    actuation_delay: tuple = (0.02, 0.06)  # s, command -> motors
    perception_delay: tuple = (0.02, 0.05)  # s, world -> observation
    obs_pos_std: tuple = (0.0, 0.6)  # cm
    obs_angle_std: tuple = (0.0, 0.02)  # rad
    obs_vel_std: tuple = (0.0, 4.0)  # cm/s

    def sample(self, rng: np.random.Generator) -> dict:
        out = {}
        for f in fields(self):
            if f.name == "enabled":
                continue
            lo, hi = getattr(self, f.name)
            if self.enabled:
                out[f.name] = float(rng.uniform(lo, hi))
            else:
                out[f.name] = 0.0 if f.name.startswith("obs_") else 0.5 * (lo + hi)
        return out


@dataclass
class RewardWeights:
    car_progress: float = 5.0  # potential shaping on car -> ball gap
    touch: float = 1.0  # one-time bonus for the first touch
    align: float = 2.0  # pre-touch: approach along the ball -> goal line (get behind the ball)
    ball_progress: float = 10.0  # potential shaping on ball -> opponent goal (after touch)
    aim_success: float = 3.0  # stage 1: ball leaves heading at the goal
    goal: float = 10.0
    conceded: float = -5.0
    out_of_bounds: float = -1.0
    time: float = 0.005  # per step
    steer_rate: float = 0.05  # * (change in steering command)^2
    throttle_rate: float = 0.02  # * (change in throttle command)^2
    gamma: float = 0.99  # must match PPO gamma for potential-based shaping


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Vehicle:
    """Kinematic bicycle car with motor lag and a rate-limited steering servo."""

    def __init__(self, space, params, pos, heading, motor_scale=1.0, servo_scale=1.0):
        self.params = params
        self.motor_scale = motor_scale
        self.servo_scale = servo_scale
        self.speed = 0.0
        self.steer_angle = 0.0
        self.body = pymunk.Body(CAR_MASS, pymunk.moment_for_box(CAR_MASS, CAR_SIZE))
        self.body.position = tuple(pos)
        self.body.angle = heading
        self.shape = pymunk.Poly.create_box(self.body, CAR_SIZE)
        self.shape.friction = 0.5
        self.shape.elasticity = 0.2
        self.shape.filter = pymunk.ShapeFilter(categories=CAT_CAR, mask=CAT_BALL | CAT_WALL | CAT_CAR)
        space.add(self.body, self.shape)

    @property
    def v_max(self) -> float:
        return self.params.v_max * self.motor_scale

    def command(self, throttle: float, steer: float, dt: float):
        p = self.params
        target = throttle * self.v_max
        speeding_up = abs(target) > abs(self.speed) and target * self.speed >= 0
        limit = (p.accel * self.motor_scale if speeding_up else p.brake) * dt
        self.speed += float(np.clip(target - self.speed, -limit, limit))

        rate = math.radians(p.servo_rate_dps) * self.servo_scale * dt
        want = steer * math.radians(p.max_steer_deg)
        self.steer_angle += float(np.clip(want - self.steer_angle, -rate, rate))

        heading = self.body.angle
        self.body.velocity = (self.speed * math.cos(heading), self.speed * math.sin(heading))
        self.body.angular_velocity = self.speed / p.wheelbase * math.tan(self.steer_angle)

    def sync(self):
        """Read back speed after collisions so a blocked car really slows down."""
        heading = self.body.angle
        forward = pymunk.Vec2d(math.cos(heading), math.sin(heading))
        self.speed = float(np.clip(self.body.velocity.dot(forward), -self.v_max, self.v_max))


class RocketLeagueEnv(gym.Env):
    """Single-agent Rocket League task. Action = [throttle, steer], both in [-1, 1]."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: int = 0,
        review_prob: float | tuple = DEFAULT_REVIEW_PROBS,
        review_weights: tuple = DEFAULT_REVIEW_WEIGHTS,
        randomization: Randomization | None = None,
        vehicle: VehicleParams | None = None,
        rewards: RewardWeights | None = None,
        control_dt: float = 0.05,
        substeps: int = 10,
        observe_ball_height: bool = True,
        max_steps: dict | None = None,
        aim_delay_steps: int = 3,
    ):
        super().__init__()
        self.rand = randomization or Randomization()
        self.vehicle_params = vehicle or VehicleParams()
        self.w = rewards or RewardWeights()
        self.dt = control_dt
        self.substeps = substeps
        self.h = control_dt / substeps
        self.observe_ball_height = observe_ball_height
        self.max_steps = max_steps or DEFAULT_MAX_STEPS
        self.aim_delay_steps = aim_delay_steps
        self.review_prob = review_prob
        self.review_weights = review_weights
        self.stage = 0
        self.set_stage(stage)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(OBS_DIM,), dtype=np.float32)
        self.space = None

    # ------------------------------------------------------------------ API
    def set_stage(self, stage: int):
        if not 0 <= stage < len(STAGE_NAMES):
            raise ValueError(f"stage must be in [0, {len(STAGE_NAMES) - 1}]")
        self.stage = int(stage)

    def get_stage(self) -> int:
        return self.stage

    def _review_prob(self) -> float:
        rp = self.review_prob
        return float(rp[self.stage]) if isinstance(rp, (tuple, list, np.ndarray)) else float(rp)

    def _sample_review_stage(self) -> int:
        """Pick which earlier stage to rehearse, weighted toward the more fundamental ones
        (``review_weights``) rather than uniformly at random."""
        weights = np.asarray(self.review_weights[:self.stage], dtype=float)
        return int(self.np_random.choice(self.stage, p=weights / weights.sum()))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        stage = self.stage
        if options and "stage" in options:
            stage = int(options["stage"])
        elif stage > 0 and self.np_random.random() < self._review_prob():
            stage = self._sample_review_stage()
        self.active_stage = stage
        self.p = self.rand.sample(self.np_random)
        self._build_world()

        self.step_count = 0
        self.t = 0.0
        self.touched = False
        self.touch_step = None
        self.in_contact = False
        self.pre_rel_vel = pymunk.Vec2d(0, 0)
        self.act_hist = [(-1e9, np.zeros(2))]
        self.prev_actions = deque([np.zeros(2)] * ACTION_HISTORY, maxlen=ACTION_HISTORY)
        self.phi_car = -self._car_target_gap() / FIELD_WIDTH
        self.phi_ball = -self._dist(self._ball_pos(), OPP_GOAL) / FIELD_WIDTH
        self.phi_align = self._align_potential()
        self.obs_delay_steps = int(round(self.p["perception_delay"] / self.dt))
        self.snapshots = deque(maxlen=8)
        self.snapshots.append(self._snapshot())
        return self._observation(), self._info(False)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.act_hist.append((self.t, action.copy()))
        just_touched = False
        for j in range(self.substeps):
            applied = self._delayed_action(self.t + j * self.h)
            self.car.command(applied[0], applied[1], self.h)
            if self.defender is not None:
                self.defender.command(*self._defender_action(), self.h)
            self._update_ball_filter()
            self.pre_rel_vel = self.car.body.velocity - self.ball.body.velocity
            self.space.step(self.h)
            self.car.sync()
            if self.defender is not None:
                self.defender.sync()
            just_touched |= self._post_physics()
        self.t += self.dt
        self.step_count += 1
        self.act_hist = self.act_hist[-6:]

        reward, terminated, success, flags = self._reward_and_done(action, just_touched)
        truncated = (not terminated) and self.step_count >= self.max_steps[self.active_stage]
        self.prev_actions.appendleft(action.copy())
        self.snapshots.append(self._snapshot())
        return self._observation(), float(reward), terminated, truncated, self._info(success, **flags)

    # ---------------------------------------------------------------- world
    def _build_world(self):
        rng = self.np_random
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        gd, fw, fh, sw, gh = GOAL_DEPTH, FIELD_WIDTH, FIELD_HEIGHT, SIDE_WALL, GOAL_HEIGHT
        segs = [
            ((gd, 0), (fw, 0)), ((fw, fh), (gd, fh)),
            ((fw, 0), (fw, sw)), ((fw, sw), (fw + gd, sw)), ((fw + gd, sw), (fw + gd, gh + sw)),
            ((fw, gh + sw), (fw + gd, sw + gh)), ((fw, gh + sw), (fw, fh)),
            ((gd, sw), (gd, 0)), ((gd, sw), (0, sw)), ((0, sw), (0, gh + sw)),
            ((gd, gh + sw), (0, sw + gh)), ((gd, gh + sw), (gd, fh)),
        ]
        for a, b in segs:
            s = pymunk.Segment(self.space.static_body, a, b, 1.5)
            s.friction = 0.3
            s.elasticity = 0.9
            s.filter = pymunk.ShapeFilter(categories=CAT_WALL)
            self.space.add(s)

        stage = self.active_stage
        margin = 35.0
        ball_xy = np.array([rng.uniform(GOAL_DEPTH + 60, FIELD_WIDTH - 60),
                            rng.uniform(margin, FIELD_HEIGHT - margin)])
        ball_v = np.zeros(2)
        if stage >= 3:
            ang = rng.uniform(0, 2 * math.pi)
            ball_v = rng.uniform(20, 80) * np.array([math.cos(ang), math.sin(ang)])
        car_xy = ball_xy
        for _ in range(50):
            car_xy = np.array([rng.uniform(GOAL_DEPTH + 25, FIELD_WIDTH - 25),
                               rng.uniform(25, FIELD_HEIGHT - 25)])
            if self._dist(car_xy, ball_xy) > 60:
                break
        self.car = Vehicle(self.space, self.vehicle_params, car_xy, rng.uniform(-math.pi, math.pi),
                           self.p["motor_scale"], self.p["servo_scale"])

        self.defender = None
        if stage >= 4:
            self.defender_mode = "keeper" if rng.random() < 0.5 else "chaser"
            self.defender_speed = rng.uniform(0.4, 0.8)
            start = (FIELD_WIDTH - 45, FIELD_HEIGHT / 2) if self.defender_mode == "keeper" \
                else (FIELD_WIDTH - 90, rng.uniform(margin, FIELD_HEIGHT - margin))
            self.defender = Vehicle(self.space, self.vehicle_params, start, math.pi,
                                    self.p["motor_scale"] * 0.85, self.p["servo_scale"])

        mass = self.p["ball_mass"]
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, BALL_RADIUS))
        body.position = tuple(ball_xy)
        body.velocity = tuple(ball_v)
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.friction = self.p["ball_friction"]
        shape.elasticity = self.p["ball_elasticity"]
        self.ball_mask = CAT_WALL | CAT_CAR
        shape.filter = pymunk.ShapeFilter(categories=CAT_BALL, mask=self.ball_mask)
        self.space.add(body, shape)
        self.ball = type("Ball", (), {"body": body, "shape": shape})()
        self.ball_h = 0.0  # height of the ball above its resting position
        self.ball_vh = 0.0

    def _update_ball_filter(self):
        if self.ball_h < CAR_HEIGHT:
            mask = CAT_WALL | CAT_CAR
        elif self.ball_h < WALL_HEIGHT:
            mask = CAT_WALL
        else:
            mask = 0
        if mask != self.ball_mask:
            self.ball_mask = mask
            self.ball.shape.filter = pymunk.ShapeFilter(categories=CAT_BALL, mask=mask)

    def _post_physics(self) -> bool:
        """Ball vertical motion, rolling friction and contact detection. True on a new agent touch."""
        body = self.ball.body
        body.angular_velocity = 0.0  # treat the ball as non-spinning
        if self.ball_h > 0 or self.ball_vh > 0:
            self.ball_vh -= GRAVITY * self.h
            self.ball_h += self.ball_vh * self.h
            if self.ball_h <= 0:
                self.ball_h = 0.0
                self.ball_vh = -self.ball_vh * self.p["floor_restitution"] if self.ball_vh < -15 else 0.0
        else:
            speed = body.velocity.length
            if speed > 0:
                body.velocity = body.velocity * (max(0.0, speed - self.p["ball_decel"] * self.h) / speed)

        touching = []
        body.each_arbiter(lambda arb: touching.append(self.car.shape in arb.shapes))
        contact = self.ball_h < CAR_HEIGHT and any(touching)
        new_touch = contact and not self.in_contact
        self.in_contact = contact
        if new_touch:
            normal = self.ball.body.position - self.car.body.position
            if normal.length > 0:
                normal = normal.normalized()
            closing = max(0.0, self.pre_rel_vel.dot(normal))  # measured before the solver resolved it
            self.ball_vh += self.p["pop_coeff"] * closing  # bumper wedge pops the ball up
            if not self.touched:
                self.touched = True
                self.touch_step = self.step_count
                return True
        return False

    def _delayed_action(self, t: float) -> np.ndarray:
        cutoff = t - self.p["actuation_delay"]
        applied = self.act_hist[0][1]
        for t_issue, act in self.act_hist:
            if t_issue <= cutoff:
                applied = act
        return applied

    def _defender_action(self):
        d = self.defender
        ball = self._ball_pos()
        if self.defender_mode == "keeper":
            target = np.array([FIELD_WIDTH - 45, np.clip(ball[1], GOAL_Y_MIN + 10, GOAL_Y_MAX - 10)])
        else:
            target = ball
        delta = target - np.array(d.body.position)
        dist = float(np.linalg.norm(delta))
        if dist < 8:
            return 0.0, 0.0
        err = _wrap(math.atan2(delta[1], delta[0]) - d.body.angle)
        throttle = min(dist / 40.0, 1.0) * self.defender_speed
        if math.cos(err) < 0:  # target behind: reverse toward it
            return -throttle, float(np.clip(-_wrap(err - math.pi) / 0.5, -1, 1))
        return throttle, float(np.clip(err / 0.5, -1, 1))

    # ------------------------------------------------------------ utilities
    @staticmethod
    def _dist(a, b) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def _ball_pos(self) -> np.ndarray:
        return np.array(self.ball.body.position)

    def _lead_time(self, rel_pos: np.ndarray, rel_vel: np.ndarray, pursuer_speed: float) -> float:
        """Time for a pursuer at the origin, moving at ``pursuer_speed``, to intercept a target
        at ``rel_pos`` holding constant velocity ``rel_vel``. Solves the standard pursuit
        quadratic ``|rel_pos + rel_vel t| = pursuer_speed * t`` for the smallest positive root,
        and falls back to ``LEAD_TIME_MAX`` when there is no real intercept (e.g. the target is
        outrunning the assumed pursuit speed) so the shaping term stays bounded either way."""
        a = rel_vel @ rel_vel - pursuer_speed ** 2
        b = 2.0 * (rel_pos @ rel_vel)
        c = rel_pos @ rel_pos
        if abs(a) < 1e-9:
            t = -c / b if abs(b) > 1e-9 else 0.0
        else:
            disc = b * b - 4 * a * c
            if disc < 0:
                return LEAD_TIME_MAX
            sq = math.sqrt(disc)
            positive = [r for r in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if r > 0]
            t = min(positive) if positive else 0.0
        return float(np.clip(t, 0.0, LEAD_TIME_MAX))

    def _intercept_point(self) -> np.ndarray:
        """Predicted ball position when the car could first reach it, assuming the ball holds
        its current velocity. Equals the ball's current position when it isn't moving (stages
        0-2), and leads a moving ball (stages 3-4) so the approach/align shaping below aims the
        car at where the ball will be rather than where it was measured. Reward shaping only:
        the physics and the policy's own observations still use the ball's true position."""
        car_pos, ball = np.array(self.car.body.position), self._ball_pos()
        t = self._lead_time(ball - car_pos, np.array(self.ball.body.velocity), self.car.v_max * LEAD_SPEED_FRAC)
        point = ball + np.array(self.ball.body.velocity) * t
        return np.array([np.clip(point[0], GOAL_DEPTH, FIELD_WIDTH), np.clip(point[1], 0.0, FIELD_HEIGHT)])

    def _car_target_gap(self) -> float:
        return max(0.0, self._dist(self.car.body.position, self._intercept_point()) - CAR_SIZE[0] / 2 - BALL_RADIUS)

    def _align_potential(self) -> float:
        """0 when the car is exactly behind the intercept point on the (point -> goal) line,
        -1 when opposite. Using the intercept point rather than the ball's raw position means
        the car learns to get on the far side of where a moving ball will be, not where it is."""
        point = self._intercept_point()
        to_goal, to_point = OPP_GOAL - point, point - np.array(self.car.body.position)
        err = _wrap(math.atan2(to_goal[1], to_goal[0]) - math.atan2(to_point[1], to_point[0]))
        return -abs(err) / math.pi

    def _aimed_at_goal(self) -> bool:
        p, v = self._ball_pos(), np.array(self.ball.body.velocity)
        if v[0] < 40:
            return False
        y = p[1] + v[1] * (FIELD_WIDTH - p[0]) / v[0]
        return GOAL_Y_MIN + BALL_RADIUS < y < GOAL_Y_MAX - BALL_RADIUS

    # ------------------------------------------------------- reward and done
    def _reward_and_done(self, action, just_touched):
        w, stage = self.w, self.active_stage
        ball, ball_v = self._ball_pos(), np.array(self.ball.body.velocity)
        r = -w.time

        phi_car = -self._car_target_gap() / FIELD_WIDTH
        if (not self.touched) or np.linalg.norm(ball_v) < 30:  # don't punish launching the ball away
            r += w.car_progress * (w.gamma * phi_car - self.phi_car)
        self.phi_car = phi_car

        phi_align = self._align_potential()
        if stage >= 1 and not self.touched:
            r += w.align * (w.gamma * phi_align - self.phi_align)
        self.phi_align = phi_align

        phi_ball = -self._dist(ball, OPP_GOAL) / FIELD_WIDTH
        if stage >= 1 and self.touched:  # only credit motion after the agent has touched it
            r += w.ball_progress * (w.gamma * phi_ball - self.phi_ball)
        self.phi_ball = phi_ball

        if just_touched:
            r += w.touch
        d = action - self.prev_actions[0]
        r -= w.throttle_rate * d[0] ** 2 + w.steer_rate * d[1] ** 2

        in_mouth = GOAL_Y_MIN < ball[1] < GOAL_Y_MAX and self.ball_h < CROSSBAR_HEIGHT
        scored = ball[0] > FIELD_WIDTH and in_mouth
        conceded = ball[0] < GOAL_DEPTH and in_mouth
        oob = not (0 <= ball[0] <= FIELD_WIDTH + GOAL_DEPTH and 0 <= ball[1] <= FIELD_HEIGHT)

        terminated = success = False
        if scored:
            r += w.goal
            terminated = success = True
        elif conceded:
            r += w.conceded
            terminated = True
        elif oob:
            r += w.out_of_bounds
            terminated = True
        elif stage == 0 and self.touched:
            terminated = success = True
        elif stage == 1 and self.touched and self.step_count - self.touch_step >= self.aim_delay_steps:
            terminated = True
            success = self._aimed_at_goal()
            if success:
                r += w.aim_success
        return r, terminated, success, {"scored": scored, "conceded": conceded, "out_of_bounds": oob}

    # ---------------------------------------------------------- observation
    def _snapshot(self) -> dict:
        c, b, d = self.car.body, self.ball.body, self.defender
        return {
            "car_pos": np.array(c.position), "heading": c.angle, "car_vel": np.array(c.velocity),
            "yaw": c.angular_velocity, "ball_pos": np.array(b.position), "ball_vel": np.array(b.velocity),
            "ball_h": self.ball_h, "ball_vh": self.ball_vh,
            "def_pos": np.array(d.body.position) if d else np.zeros(2),
            "def_vel": np.array(d.body.velocity) if d else np.zeros(2),
            "def_present": d is not None,
        }

    def _observation(self) -> np.ndarray:
        s = self.snapshots[max(0, len(self.snapshots) - 1 - self.obs_delay_steps)]
        rng, p = self.np_random, self.p

        def noisy(x, std):
            return x + rng.normal(0.0, std, size=np.shape(x)) if std > 0 else x

        car_pos = noisy(s["car_pos"], p["obs_pos_std"])
        heading = float(noisy(s["heading"], p["obs_angle_std"]))
        car_vel = noisy(s["car_vel"], p["obs_vel_std"])
        ball_pos = noisy(s["ball_pos"], p["obs_pos_std"])
        ball_vel = noisy(s["ball_vel"], p["obs_vel_std"])
        fwd = np.array([math.cos(heading), math.sin(heading)])
        right = np.array([-math.sin(heading), math.cos(heading)])

        def to_car(v):
            return np.array([v @ fwd, v @ right])

        if s["def_present"]:
            defender = np.concatenate([to_car(noisy(s["def_pos"], p["obs_pos_std"]) - car_pos) / POS_SCALE,
                                       to_car(s["def_vel"] - car_vel) / VEL_SCALE, [1.0]])
        else:
            defender = np.zeros(5)
        height = [s["ball_h"] / H_SCALE, s["ball_vh"] / VH_SCALE] if self.observe_ball_height else [0.0, 0.0]
        obs = np.concatenate([
            to_car(ball_pos - car_pos) / POS_SCALE,
            to_car(ball_vel - car_vel) / VEL_SCALE,
            height,
            to_car(car_vel) / CAR_VEL_SCALE,
            [s["yaw"] / YAW_SCALE],
            [math.sin(heading), math.cos(heading)],
            to_car(OPP_GOAL - car_pos) / GOAL_SCALE,
            to_car(OWN_GOAL - car_pos) / GOAL_SCALE,
            defender,
            np.concatenate(list(self.prev_actions)),
        ])
        return np.clip(obs, -3.0, 3.0).astype(np.float32)

    def _info(self, success, **flags) -> dict:
        return {"is_success": bool(success), "stage": self.active_stage, "touched": self.touched,
                "actuation_delay": self.p["actuation_delay"], "perception_delay": self.p["perception_delay"],
                **flags}
