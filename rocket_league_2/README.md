# Rocket League 2

Teaching cars soccer

Uses [ROS 2: Jazzy](https://docs.ros.org/en/jazzy/index.html)

For more information, check out the [wiki](https://github.com/purdue-arc/rocket_league_2/wiki/)

[ROS Workshop](https://ivory-sale-974.notion.site/ARC-ROS-Workshop-2d26d5bcdd69496996806ccf8e5a011b) for onboarding

## RL training

`rktl_simulator/rktl_simulator/car_env.py` is a headless [Gymnasium](https://gymnasium.farama.org/)
environment used to train the driving policy with PPO, separate from the ROS/pygame simulator used
for manual play and integration testing. It models the car as a bicycle with motor/steering lag, the
ball as a 2.5D body (pymunk in the plane, height integrated separately so a hit can pop it off the
floor), and randomizes physics, actuation delay and perception delay/noise each episode so the policy
doesn't overfit to one exact simulator.

Training follows a 5-stage curriculum, advanced automatically once a stage's success rate clears its
gate, with earlier stages kept in rotation ("review") afterwards so the policy doesn't forget them:

| stage | name | task |
|---|---|---|
| 0 | `reach` | drive to a stationary ball |
| 1 | `aim` | touch a stationary ball so it heads toward the goal |
| 2 | `score_static` | score on a stationary ball |
| 3 | `score_rolling` | score on a moving ball |
| 4 | `score_defender` | score against a scripted defender |

Run training and evaluation from `rktl_simulator/`:

```sh
python -m rktl_simulator.train_ppo --steps 12000000 --n-envs 8 --out runs/ppo1
python -m rktl_simulator.evaluate runs/ppo1/final.zip --episodes 100
```

Training output (`--out`) is git-ignored; it's reproducible from a run and not meant to be committed.

This environment is the fast, cheap-to-iterate testbed for the algorithm, reward shaping, and
curriculum. The plan is to port a validated recipe to a full 3D physics sim (Isaac Sim/Isaac Lab)
once it reliably clears all 5 stages here.
