"""Training callbacks: task-metric logging + best-by-success checkpoint."""
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateCallback(BaseCallback):
    """Log rollout/success_rate and victims_found_frac; optionally save the best policy by success (not reward)."""

    def __init__(self, window: int = 100, best_save_dir=None,
                 vecnormalize=None, verbose: int = 0):
        super().__init__(verbose)
        self._succ: deque = deque(maxlen=window)
        self._vf: deque = deque(maxlen=window)
        self._cov: deque = deque(maxlen=window)
        self.best_save_dir = Path(best_save_dir) if best_save_dir else None
        self.vecnormalize = vecnormalize
        self.best = -1.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:  # Monitor only adds "episode" on episode end
                continue
            outcome = ep.get("outcome", info.get("outcome"))
            self._succ.append(1.0 if outcome == "success" else 0.0)
            total = ep.get("total_victims", info.get("total_victims", 0)) or 0
            found = ep.get("victims_found", info.get("victims_found", 0))
            self._vf.append(found / total if total > 0 else 0.0)
            self._cov.append(float(ep.get("area_explored", info.get("area_explored", 0.0))))
        if self._succ:
            sr = float(np.mean(self._succ))
            self.logger.record("rollout/success_rate", sr)
            self.logger.record("rollout/victims_found_frac", float(np.mean(self._vf)))
            self.logger.record("rollout/coverage_frac", float(np.mean(self._cov)))
            # Save best-by-success once a full window of episodes has accumulated.
            if (self.best_save_dir is not None
                    and len(self._succ) == self._succ.maxlen
                    and sr > self.best):
                self.best = sr
                self.best_save_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(self.best_save_dir / "best_model")
                if self.vecnormalize is not None:
                    self.vecnormalize.save(
                        str(self.best_save_dir / "best_vecnormalize.pkl"))
                if self.verbose:
                    print(f"[best] success_rate={sr:.3f} @ {self.num_timesteps} "
                          f"steps -> saved best_model", flush=True)
        return True


class EntCoefAnneal(BaseCallback):
    """Linearly anneal PPO's ent_coef from start to end over total_timesteps (explore early, commit late)."""

    def __init__(self, start: float = 0.05, end: float = 0.01,
                 total_timesteps: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total = max(1, total_timesteps)

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total)
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True
