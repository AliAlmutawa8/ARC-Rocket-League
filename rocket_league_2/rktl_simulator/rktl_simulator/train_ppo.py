"""Train the car with PPO and a success-gated curriculum.

    python -m rktl_simulator.train_ppo --steps 3000000 --n-envs 8 --out runs/ppo1

The curriculum advances a stage when the success rate over the last ``window``
episodes *of the current stage* passes that stage's threshold. Earlier stages keep
being sampled (``review_prob`` in the env) so the policy doesn't forget them.
"""
from __future__ import annotations

import argparse
import math
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rktl_simulator.car_env import STAGE_NAMES, RocketLeagueEnv

# Success rate needed to leave each stage (the last stage has none). "aim" (1) is held to a
# higher bar than the rest: it's the fundamental positioning skill everything after it builds
# on (approach + get-behind-the-ball), so it's worth the extra training time to lock it in
# solidly rather than advance on a shaky ~55% and rely on review alone to patch it up later.
STAGE_THRESHOLDS = (0.90, 0.75, 0.65, 0.50, None)


class CurriculumCallback(BaseCallback):
    def __init__(self, window: int = 200, thresholds=STAGE_THRESHOLDS, start_stage: int = 0):
        super().__init__()
        self.window = window
        self.thresholds = thresholds
        self.stage = start_stage
        self.results = deque(maxlen=window)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and info.get("stage") == self.stage:  # ignore review-stage episodes
                self.results.append(float(info["is_success"]))
        rate = float(np.mean(self.results)) if self.results else 0.0
        self.logger.record("curriculum/stage", self.stage)
        self.logger.record("curriculum/success_rate", rate)
        if self.num_timesteps % 40960 < self.training_env.num_envs:
            print(f"[curriculum] step {self.num_timesteps}: stage {self.stage} "
                  f"success {rate:.2f} over {len(self.results)} eps", flush=True)
        threshold = self.thresholds[self.stage]
        if threshold and len(self.results) == self.window and rate >= threshold:
            self.stage += 1
            self.results.clear()
            self.training_env.env_method("set_stage", self.stage)
            print(f"[curriculum] step {self.num_timesteps}: success {rate:.2f} -> "
                  f"stage {self.stage} ({STAGE_NAMES[self.stage]})")
        return True


class EntropyGuardCallback(BaseCallback):
    """Guards against two entropy failure modes seen in practice with a *fixed* ent_coef:
    too low (0.005) collapsed the action std to ~0.1-0.16 within ~150k steps, killing
    exploration before "aim" was learned; too high (0.01) avoided that but let std grow
    *unbounded* later - it climbed from ~0.25 past 1.8 over a 12M-step run, once the policy
    hit score_rolling and stopped improving fast enough to out-pull the entropy bonus, so
    most of training on the harder stages was spent on near-random, heavily-clipped actions.

    Fixes both by annealing ent_coef down over training (protects early exploration, then
    favors precision later) and hard-clamping log_std each step as a backstop that holds
    regardless of how well that anneal is tuned.
    """

    def __init__(self, total_steps: int, ent_coef_start: float, ent_coef_end: float = 0.002,
                std_min: float = 0.05, std_max: float = 1.2):
        super().__init__()
        self.total_steps = total_steps
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.log_std_min = math.log(std_min)
        self.log_std_max = math.log(std_max)

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total_steps)
        self.model.ent_coef = self.ent_coef_start + frac * (self.ent_coef_end - self.ent_coef_start)
        with torch.no_grad():
            self.model.policy.log_std.clamp_(self.log_std_min, self.log_std_max)
        self.logger.record("entropy_guard/ent_coef", self.model.ent_coef)
        self.logger.record("entropy_guard/std_mean", float(self.model.policy.log_std.exp().mean()))
        return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-stage", type=int, default=0)
    ap.add_argument("--ent-coef", type=float, default=0.01, help="starting PPO entropy bonus coefficient")
    ap.add_argument("--ent-coef-end", type=float, default=0.002,
                    help="entropy bonus coefficient annealed down to by the end of training")
    ap.add_argument("--log-std-init", type=float, default=0.0, help="initial log std of the action Gaussian")
    ap.add_argument("--std-min", type=float, default=0.05, help="hard floor on the action std, every step")
    ap.add_argument("--std-max", type=float, default=1.2, help="hard ceiling on the action std, every step")
    ap.add_argument("--thresholds", type=float, nargs=4, default=None,
                    help="override the 4 success-rate gates (stage 4 has none)")
    args = ap.parse_args(argv)

    env = make_vec_env(RocketLeagueEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"stage": args.start_stage})
    try:
        import tensorboard  # noqa: F401
        tb_log = args.out
    except ImportError:
        tb_log = None  # curriculum progress is still printed to stdout
    model = PPO(
        "MlpPolicy", env, seed=args.seed, device="cpu", verbose=1, tensorboard_log=tb_log,
        n_steps=512, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef,
        learning_rate=3e-4,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "log_std_init": args.log_std_init},
    )
    thresholds = tuple(args.thresholds) + (None,) if args.thresholds else STAGE_THRESHOLDS
    callbacks = [CurriculumCallback(thresholds=thresholds, start_stage=args.start_stage),
                 EntropyGuardCallback(args.steps, args.ent_coef, args.ent_coef_end, args.std_min, args.std_max),
                 CheckpointCallback(100_000 // args.n_envs, args.out + "/checkpoints")]
    model.learn(args.steps, callback=callbacks)
    model.save(args.out + "/final")


if __name__ == "__main__":
    main()
