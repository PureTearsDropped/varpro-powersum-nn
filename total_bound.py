#!/usr/bin/env python3
"""指数の上限 |w|≤6 を外したら、全域化された算術は何を買うか。

12法則ベンチで見張りが一度も発火しなかったのは、**上限が算術の代わりに仕事をしていた**からだった。
上限が算術を守っているのなら、算術が全域なら上限を緩められるはずである。試す。
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
import separable_fit as sf
from total_arith import total_power

_orig = sf.power_unit
CNT = {"inf": 0, "under": 0, "calls": 0}
def counting(x, w):
    v = _orig(x, w); CNT["calls"] += 1
    va = np.asarray(v, float); xa = np.abs(np.asarray(x, float))
    if not np.all(np.isfinite(va)): CNT["inf"] += 1
    if np.any((va == 0.0) & (xa != 0.0)): CNT["under"] += 1
    return v

_src = open("benchmark12.py").read().split("for zeros in (False, True):")[0]
_ns = {}; exec(compile(_src, "benchmark12.py", "exec"), _ns)
LAWS, check = _ns["LAWS"], _ns["check"]

print("=" * 92)
print("指数の上限 `bound` を上げていくと、どうなるか   （3 seed × 12 法則、零点なし）")
print("=" * 92)
print(f"  {'bound':>8}{'現行 同定':>12}{'**全域** 同定':>16}{'現行で Inf/NaN が出た':>22}{'偽ゼロ':>10}")
for bound in (6.0, 12.0, 30.0, 60.0):
    res = {}
    for mode in ("現行", "全域"):
        sf.power_unit = counting if mode == "現行" else total_power
        if mode == "現行": CNT.update({"inf": 0, "under": 0, "calls": 0})
        ok = tot = 0
        for seed in (0, 1, 2):
            rng = np.random.default_rng(20260716 + seed)
            for name, truth, vn, tr, tt, zi in LAWS:
                n = 400
                X = np.c_[tuple(rng.uniform(a, b, n) for a, b in tr)]
                y = truth(X) * np.exp(rng.normal(0, 0.01, n))
                f = sf.fit_powersum(X, y, Kmax=3, seeds=15, bound=bound)
                tot += 1; ok += bool(check(f, tt))
        res[mode] = (ok, tot)
        if mode == "現行": snap = dict(CNT)
    sf.power_unit = _orig
    print(f"  {bound:>8.0f}{f'{res[chr(29694)+chr(34892)][0]}/{res[chr(29694)+chr(34892)][1]}':>12}"
          f"{f'**{res[chr(20840)+chr(22495)][0]}/{res[chr(20840)+chr(22495)][1]}**':>16}"
          f"{f'{snap[chr(105)+chr(110)+chr(102)]:,} / {snap[chr(99)+chr(97)+chr(108)+chr(108)+chr(115)]:,} 回':>22}{snap['under']:>10,}")
