import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pymunk")

from rktl_simulator.car_env import (  # noqa: E402
    DEFAULT_REVIEW_PROBS, OBS_DIM, Randomization, RocketLeagueEnv,
)


def make(stage=0, **kw):
    kw.setdefault("review_prob", 0.0)
    return RocketLeagueEnv(stage=stage, **kw)


def no_rand():
    return Randomization(enabled=False)


def test_spaces_and_rollout_all_stages():
    for stage in range(5):
        env = make(stage)
        obs, _ = env.reset(seed=stage)
        assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
        for _ in range(60):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all() and np.isfinite(r)
            if term or trunc:
                break
        assert info["stage"] == stage


def test_seed_is_deterministic():
    runs = []
    for _ in range(2):
        env = make(3)
        obs, _ = env.reset(seed=7)
        trace = [obs]
        rng = np.random.default_rng(0)
        for _ in range(30):
            obs, *_ = env.step(rng.uniform(-1, 1, 2))
            trace.append(obs)
        runs.append(np.array(trace))
    assert np.allclose(runs[0], runs[1])


def test_car_dynamics_follow_commands():
    env = make(0, randomization=no_rand())
    env.reset(seed=0)
    env.car.body.position = (200, 150)
    env.car.body.angle = 0.0
    env.ball.body.position = (400, 40)  # keep the ball out of the way
    for _ in range(20):
        env.step(np.array([1.0, 0.0]))
    assert env.car.speed > 0.9 * env.car.v_max  # reaches top speed
    y0 = env.car.body.position.y
    for _ in range(5):
        env.step(np.array([1.0, 1.0]))
    assert env.car.body.position.y > y0  # positive steer -> +y ("right" on screen)
    assert env.car.steer_angle <= math.radians(30) + 1e-9  # steering is clamped


def test_actuation_delay_is_applied():
    env = make(0, randomization=no_rand())
    env.reset(seed=0)
    env.p["actuation_delay"] = 0.09
    env.step(np.array([1.0, 0.0]))
    assert env._delayed_action(0.0)[0] == 0.0  # command issued at t=0 not yet active
    assert env._delayed_action(0.20)[0] == 1.0  # active once the delay has passed


def test_ball_gets_height_after_hit_and_lands():
    env = make(2, randomization=no_rand())
    env.reset(seed=0)
    env.p["pop_coeff"] = 0.4
    env.ball.body.position = (200, 150)
    env.ball.body.velocity = (0, 0)
    env.car.body.position = (170, 150)
    env.car.body.angle = 0.0
    peak = 0.0
    for _ in range(12):
        env.step(np.array([1.0, 0.0]))
        peak = max(peak, env.ball_h)
    assert env.touched and peak > 0.1
    for _ in range(100):
        env.step(np.array([0.0, 0.0]))
    assert env.ball_h == 0.0  # gravity brought it back down


def test_reach_stage_succeeds_on_touch():
    env = make(0, randomization=no_rand())
    env.reset(seed=0)
    env.ball.body.position = (200, 150)
    env.car.body.position = (150, 150)
    env.car.body.angle = 0.0
    for _ in range(60):
        _, _, term, _, info = env.step(np.array([1.0, 0.0]))
        if term:
            break
    assert term and info["is_success"]


def test_goal_scores_and_terminates():
    env = make(2, randomization=no_rand())
    env.reset(seed=0)
    env.car.body.position = (100, 50)
    env.ball.body.position = (400, 152.4)
    env.ball.body.velocity = (150, 0)
    total, done = 0.0, False
    for _ in range(30):
        _, r, done, _, info = env.step(np.zeros(2))
        total += r
        if done:
            break
    assert done and info["scored"] and info["is_success"] and total > 5


def test_stage_switch_and_review():
    env = RocketLeagueEnv(stage=0, review_prob=1.0)
    env.set_stage(3)
    seen = set()
    for i in range(30):
        _, info = env.reset(seed=i)
        seen.add(info["stage"])
    assert seen <= {0, 1, 2} and len(seen) > 1  # review_prob=1 only samples earlier stages
    with pytest.raises(ValueError):
        env.set_stage(9)


def test_observation_is_egocentric():
    env = make(0, randomization=no_rand())
    env.reset(seed=0)
    env.car.body.position = (150, 150)
    env.ball.body.position = (150, 200)  # +y of the car
    obs = []
    for heading in (0.0, math.pi / 2):  # facing +x, then facing +y
        env.car.body.angle = heading
        env.snapshots.clear()
        env.snapshots.append(env._snapshot())
        obs.append(env._observation())
    assert obs[0][1] > 0.1 and abs(obs[0][0]) < 1e-3  # ball is to the right
    assert obs[1][0] > 0.1 and abs(obs[1][1]) < 1e-3  # ball is straight ahead


def test_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env
    check_env(make(3), skip_render_check=True)


def test_intercept_point_matches_ball_when_stationary():
    env = make(2, randomization=no_rand())
    env.reset(seed=0)
    env.ball.body.position = (300, 100)
    env.ball.body.velocity = (0, 0)
    env.car.body.position = (150, 150)
    assert np.allclose(env._intercept_point(), (300, 100), atol=1e-6)


def test_intercept_point_leads_a_moving_ball():
    env = make(3, randomization=no_rand())
    env.reset(seed=0)
    env.car.body.position = (100, 150)
    env.ball.body.position = (250, 150)
    env.ball.body.velocity = (40, 0)  # moving away from the car, along +x
    point = env._intercept_point()
    assert point[0] > 250  # led ahead of the ball's current position
    assert abs(point[1] - 150) < 1e-6

    stationary_gap = env._dist((100, 150), (250, 150))
    assert env._car_target_gap() > stationary_gap - 1e-6  # gap grows once the lead is included


def test_review_weights_favor_fundamental_stages():
    env = RocketLeagueEnv(stage=4, review_prob=1.0)  # always review; eligible targets are 0-3
    counts = np.zeros(4)
    for i in range(400):
        _, info = env.reset(seed=i)
        counts[info["stage"]] += 1
    shares = counts / counts.sum()
    assert shares[1] > shares[3]  # "aim" reviewed more than the least-fundamental eligible stage
    assert shares[1] > 0.30       # roughly matches its 3.0/6.8 weight share, well above uniform 25%


def test_review_prob_scales_with_current_stage():
    env = RocketLeagueEnv(stage=1)
    assert env._review_prob() == pytest.approx(DEFAULT_REVIEW_PROBS[1])
    env.set_stage(3)
    assert env._review_prob() == pytest.approx(DEFAULT_REVIEW_PROBS[3])
    assert env._review_prob() > 0  # later stages review more than earlier ones
    scalar_env = make(3, review_prob=0.2)
    assert scalar_env._review_prob() == pytest.approx(0.2)  # scalar still broadcasts uniformly
