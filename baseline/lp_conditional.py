"""Conditional rank-one editing by LP. This is a PRIVILEGED baseline.

Requires pre-specified relevant layers, behavior direction u and separate
looping/productive activations. It cannot discover a model mechanism.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

@dataclass
class LPResult:
    v: np.ndarray
    objective: float
    loop_slack: np.ndarray
    productive_slack: np.ndarray
    solver_message: str

def solve_conditional_lp(loop_acts: np.ndarray, productive_acts: np.ndarray,
                         delta: float=1.0, epsilon: float=0.05,
                         slack_penalty: float=10.0) -> LPResult:
    r = np.asarray(loop_acts, dtype=np.float64)
    p = np.asarray(productive_acts, dtype=np.float64)
    if r.ndim != 2 or p.ndim != 2 or r.shape[1] == 0 or r.shape[1] != p.shape[1]:
        raise ValueError("nonempty 2D activation feature dimensions must agree")
    if not len(r) or not len(p) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(p)):
        raise ValueError("finite, nonempty looping and productive sets required")
    if delta <= 0 or epsilon < 0 or slack_penalty <= 0:
        raise ValueError("invalid margins or penalty")
    d, nr, np_ = r.shape[1], len(r), len(p)
    # x=[vplus(d),vminus(d),slack_loop(nr),slack_prod(np_)] >= 0
    zl = np.zeros((nr,np_))
    zp = np.zeros((np_,nr))
    a_loop = np.concatenate([r,-r,-np.eye(nr),zl], axis=1)
    a_pos = np.concatenate([p,-p,zp,-np.eye(np_)], axis=1)
    a_neg = np.concatenate([-p,p,zp,-np.eye(np_)], axis=1)
    a = np.concatenate([a_loop,a_pos,a_neg],axis=0)
    b = np.concatenate([np.full(nr,-delta),np.full(np_,epsilon),np.full(np_,epsilon)])
    c = np.concatenate([np.ones(2*d),np.full(nr+np_,slack_penalty)])
    solution = linprog(c,A_ub=a,b_ub=b,bounds=(0,None),method="highs")
    if not solution.success:
        raise RuntimeError(f"LP failed: {solution.message}")
    v = solution.x[:d] - solution.x[d:2*d]
    return LPResult(v,float(solution.fun),solution.x[2*d:2*d+nr],
                    solution.x[2*d+nr:],solution.message)

def apply_rank_one(weight: np.ndarray, direction: np.ndarray, v: np.ndarray) -> np.ndarray:
    w=np.asarray(weight)
    u=np.asarray(direction,dtype=np.float64)
    v=np.asarray(v,dtype=np.float64)
    if w.ndim != 2 or u.ndim != 1 or v.ndim != 1 or w.shape != (len(u),len(v)):
        raise ValueError("rank-one edit shape mismatch")
    if not all(np.all(np.isfinite(a)) for a in (w,u,v)):
        raise ValueError("nonfinite edit")
    if not np.isclose(np.linalg.norm(u),1,rtol=1e-4,atol=1e-5):
        raise ValueError("direction must have unit norm")
    return w+np.outer(u,v)
