"""MaskableDQN: SB3 DQN with action masking.

sb3-contrib does not ship a MaskableDQN, so we build one by overriding
``predict``:
- epsilon-greedy exploration samples only legal actions (from the env's
  ``action_masks()``);
- the greedy argmax ignores masked (illegal) actions.

Note on replay: the meaning of a discrete action index depends on which
unit is acting (the env remaps legal (action, target) pairs each decision),
so off-policy replay is inherently approximate in this setting. Masking at
collection time is what guarantees the agent never *takes* illegal actions.
"""

from __future__ import annotations

import numpy as np
import torch as th

from stable_baselines3 import DQN


def _get_mask(env) -> np.ndarray | None:
    mask = None
    base = env
    if hasattr(base, "envs"):  # VecEnv
        base = base.envs[0]
    while base is not None:
        if hasattr(base, "action_masks"):
            mask = np.asarray(base.action_masks(), dtype=bool)
            break
        base = getattr(base, "env", None)
    return mask


class MaskableDQN(DQN):
    """DQN with discrete action masking; requires the env to expose
    ``action_masks()`` (cli-hsr's ``HSREnv`` does)."""

    def predict(self, observation, state=None, episode_start=None,
                deterministic: bool = False, action_masks=None):
        # accept the MaskablePPO-style kwarg; env masks are authoritative here
        mask = np.asarray(action_masks, dtype=bool) if action_masks is not None \
            else _get_mask(self.get_env())
        # VecEnv calls predict with a batched obs (n, OBS_DIM); match the shape
        batched = getattr(observation, "ndim", 1) > 1

        def _wrap(a: int) -> np.ndarray:
            scalar = np.array(a, dtype=np.int64)
            return scalar if not batched else np.array([scalar])

        if not deterministic and np.random.rand() < self.exploration_rate:
            if mask is None or mask.all():
                return super().predict(observation, state, episode_start, deterministic)
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                # terminal/non-decision state: return any action; env.step ignores it
                return _wrap(0), state
            return _wrap(int(np.random.choice(legal))), state

        with th.no_grad():
            obs_t = th.as_tensor(np.asarray(observation, dtype=np.float32))
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            q_values = self.q_net(obs_t)[0].cpu().numpy()
        if mask is not None and not mask.all():
            q_values = np.where(mask, q_values, -np.inf)
        return _wrap(int(np.argmax(q_values))), state
