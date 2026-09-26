"""Evaluate a trained PPO policy on each curriculum stage.

    python -m rktl_simulator.evaluate runs/ppo/final.zip --episodes 200
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from rktl_simulator.car_env import STAGE_NAMES, Randomization, RocketLeagueEnv


def evaluate(model, stage: int, episodes: int = 100, seed: int = 1000, randomize: bool = True) -> dict:
    env = RocketLeagueEnv(stage=stage, review_prob=0.0, randomization=Randomization(enabled=randomize))
    wins, rewards, lengths, touched = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total, steps, done = total + r, steps + 1, term or trunc
        wins.append(info["is_success"])
        touched.append(info["touched"])
        rewards.append(total)
        lengths.append(steps)
    return {"success": float(np.mean(wins)), "touch": float(np.mean(touched)),
            "reward": float(np.mean(rewards)), "steps": float(np.mean(lengths))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--no-randomize", action="store_true")
    args = ap.parse_args(argv)
    model = PPO.load(args.model, device="cpu")
    print(f"{'stage':<16}{'success':>9}{'touch':>8}{'reward':>9}{'steps':>8}")
    for stage, name in enumerate(STAGE_NAMES):
        m = evaluate(model, stage, args.episodes, randomize=not args.no_randomize)
        print(f"{stage} {name:<14}{m['success']:>9.2f}{m['touch']:>8.2f}{m['reward']:>9.2f}{m['steps']:>8.0f}")


if __name__ == "__main__":
    main()
