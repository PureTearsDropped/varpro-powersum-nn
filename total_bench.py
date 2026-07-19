#!/usr/bin/env python3
"""**全域化された算術**（溢れ→±MAX／潰れ→±MIN／a/0=0）を差し込んで、12法則ベンチをそのまま回す。

比べるのは3つ:
  現行     power_unit（0^w=0、だが np.power は Inf にも 0 にも化ける）
  **全域** total_power（溢れ→±MAX、潰れ→+MIN。**0 は本物の 0 からしか出ない**）
そして、`isfinite` の見張りが **実際に何回発火したか** を数える。
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
import separable_fit as sf
from total_arith import total_power, MAX, MIN

GUARD = {"fire": 0, "calls": 0, "inf": 0, "under": 0}
_orig_power_unit = sf.power_unit

def counting_power_unit(x, w):
    """現行の実装。ただし **見張りが何を見ているか** を数える。"""
    v = _orig_power_unit(x, w)
    GUARD["calls"] += 1
    xa = np.abs(np.asarray(x, float))
    if not np.all(np.isfinite(v)):
        GUARD["inf"] += 1
    if np.any((np.asarray(v, float) == 0.0) & (xa != 0.0)):   # **潰れ由来の偽の 0**
        GUARD["under"] += 1
    return v

def total_power_unit(x, w):
    return total_power(x, w)

# ---- isfinite の見張りが発火したら数える ----
_orig_isfinite = np.isfinite
def counting_isfinite(a, *args, **kw):
    r = _orig_isfinite(a, *args, **kw)
    if np.ndim(r) > 0 and not np.all(r):
        GUARD["fire"] += 1
    return r

import importlib.util as _u
_sp = _u.spec_from_file_location("_b12", "benchmark12.py")
_src = open("benchmark12.py").read().split("for zeros in (False, True):")[0]
_ns = {}
exec(compile(_src, "benchmark12.py", "exec"), _ns)
class B: LAWS = _ns["LAWS"]; check = staticmethod(_ns["check"])

def run(mode, zeros, seed):
    if mode == "現行":
        sf.power_unit = counting_power_unit
    else:
        sf.power_unit = total_power_unit
    rng = np.random.default_rng(seed)
    n_ok = n = 0
    for name, truth, vn, tr, tt, zi in B.LAWS:
        if zeros and zi is None:
            continue
        nn = 400
        X = np.c_[tuple(rng.uniform(a, b, nn) for a, b in tr)]
        if zeros and zi is not None:
            X[:int(0.1 * nn), zi] = 0.0
        y0 = truth(X)
        # 零点があるなら、そこのノイズは加法でしかありえない（乗法は 0 を 0 のままにする）
        if np.any(np.abs(y0) < 1e-300):
            y = y0 + rng.normal(0, 0.01 * np.abs(y0).mean(), nn)
        else:
            y = y0 * np.exp(rng.normal(0, 0.01, nn))
        f = sf.fit_powersum(X, y, Kmax=3, seeds=15)
        n += 1
        n_ok += bool(B.check(f, tt))
    sf.power_unit = _orig_power_unit
    return n_ok, n

print("=" * 84)
print("**全域化された算術**で 12 法則ベンチ  （3 seed × 零点なし/あり）")
print("=" * 84)
for zeros in (False, True):
    print(f"\n  --- {'**10% の点に 0 を混ぜる**' if zeros else '零点なし'} ---")
    for mode in ("現行", "**全域**"):
        tot_ok = tot = 0
        GUARD.update({"fire": 0, "calls": 0, "inf": 0, "under": 0})
        np.isfinite = counting_isfinite if mode == "現行" else _orig_isfinite
        for s in (0, 1, 2):
            a, b = run(mode, zeros, s)
            tot_ok += a; tot += b
        np.isfinite = _orig_isfinite
        extra = ""
        if mode == "現行":
            extra = (f"   ｜ 冪の呼び出し **{GUARD['calls']:,}** 回中、"
                     f"Inf/NaN が出た **{GUARD['inf']:,}** 回、"
                     f"**潰れて偽の 0 になった {GUARD['under']:,}** 回")
        print(f"    {mode:<8} 同定 **{tot_ok}/{tot}**{extra}")
