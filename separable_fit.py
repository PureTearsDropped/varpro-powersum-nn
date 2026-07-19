#!/usr/bin/env python3
"""可分モデルの一般当てはめライブラリ。

対象: y = Phi(theta; X) @ c   … 係数 c に線形、形状 theta に非線形 (可分構造)

手続き:
  1. VARPRO       非線形 theta のみ多スタート探索。c は各評価で閉形式の最適解。
  2. 刈り込み     係数≈0 の死に項を削除 (呼び出し側の prune_fn で指定)。
  3. 丸め         theta のうち「指数型」と宣言されたものを最近接の小有理数へ (次元解析)。
                  許容幅内のもののみ丸め、以後は固定する。
  4. 再VARPRO     丸めなかった自由な非線形パラメータを、固定分を保持したまま再探索。
                  ★丸めは他のパラメータの最適点を動かすため、混在モデルではこの段が必須。
  5. 係数解       theta を固定すれば c は線形 → 閉形式の重み付き最小二乗で厳密解。

重みは relative=True で 1/|y| (乗法ノイズの最尤)、False で等重み (加法ノイズの最尤)。
bias 列は truth が 0 でも残すこと (撹乱パラメータが本命の係数の推定を守る。実測で削ると悪化)。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import f as _fdist

warnings.filterwarnings("ignore")


def weighted_lstsq(Phi: np.ndarray, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(Phi * wt[:, None], y * wt, rcond=None)
    return coef


@dataclass
class SeparableFit:
    theta: np.ndarray          # 非線形パラメータ
    coef: np.ndarray           # 線形係数 (末尾は bias 列があればその係数)
    snapped: np.ndarray        # bool: 丸めて固定された成分
    predict: Callable[[np.ndarray], np.ndarray]
    loss: float
    report: list = None        # list[SnapReport]: 各指数の判断材料（丸めなかったものは要目視）


def _varpro_core(make_basis, X, y, wt, theta_init_fn, n_theta, seeds, complex_y,
                 free_mask=None, theta_base=None, bounds=None):
    """free_mask が与えられれば、その成分のみ探索し、他は theta_base に固定。

    bounds=(lo, hi) を与えると探索範囲を制限する。冪指数には物理的な上限を課すべき:
    無制約だと r^-71 のような暴走指数が訓練ノイズを局所的に拾って過学習し (訓練損失は
    下がるので刈り込みでも検出できない)、外挿で爆発する。
    """
    if free_mask is None:
        free_mask = np.ones(n_theta, bool)
        theta_base = np.zeros(n_theta)

    def expand(free_vals):
        th = theta_base.copy(); th[free_mask] = free_vals; return th

    def solve(theta):
        Phi = make_basis(theta, X)
        if Phi is None or not np.all(np.isfinite(Phi)):
            return None, None
        return Phi, weighted_lstsq(Phi, y, wt)

    def resid(free_vals):
        Phi, coef = solve(expand(free_vals))
        if Phi is None:
            return np.full(len(y) * (2 if complex_y else 1), 1e6)
        r = (Phi @ coef - y) * wt
        return np.r_[r.real, r.imag] if complex_y else r

    best = None
    for s in range(seeds):
        rng = np.random.default_rng(s)
        th0 = theta_init_fn(rng)
        fv0 = th0[free_mask] if len(th0) == n_theta else th0
        kw = {}
        if bounds is not None:
            lo = np.asarray(bounds[0], float)[free_mask]
            hi = np.asarray(bounds[1], float)[free_mask]
            fv0 = np.clip(fv0, lo + 1e-9, hi - 1e-9)
            kw["bounds"] = (lo, hi)
        try:
            out = least_squares(resid, fv0, max_nfev=4000, **kw)
            loss = float(np.mean(np.abs(resid(out.x)) ** 2))
            if best is None or loss < best[0]:
                best = (loss, out.x)
        except Exception:
            continue
    theta = expand(best[1])
    return theta, best[0]



# ---------------------------------------------------------------- 丸めの規則
# 格子は「有理数」ではなく **ℤ[1/2] = {p/2^m}** — 二進有理数の群。
#   生成元は 2つだけ: **1（数える）** と **1/2（平方根をとる）**
#   和と差で閉じている ⟹ **群** ⟹ 次元解析が閉じる（有理数一般には無い性質）
#   代金は m = 半分にした回数。二進有理数でないものは **∞（存在しない）** ので、稠密性の問題が消える
#
# 「近い」は **ŵ の誤差棒に対して**近いこと。固定の許容幅はデータの精度を見ないので、
# 真が格子の隣（例 w=1.03）にあると、どの精度でも丸めて、定数を黙って壊す。
# 誤差棒に対してなら、精度が上がれば自動で拒否しはじめる。
#
# 実測 (dyadic_group.py / snap_by_errorbar.py):
#   真 w=1.03, 精度 1e-3:  全有理数 q≤√n → 13/12 に丸め μ=−289.98（誤差 16.8）
#                          **ℤ[1/2], m≤3 → 丸めず μ=−278.67（正直な不精密）**
#   真が格子の上 (1, 3/2, 1/2, 7/4, 1/8): 全て正しく拾う
#   実在の物理指数 17個中 **12個** が二進有理数。外れる5個は全部 分母に 3 を持つ（CFT・乱流）
#
# **これは事前分布である**: 「物理は 数える＋平方根 でできている」という賭け。的中 12/17。
# 外すとき（CFT の指数）は **拒否する**ので、嘘にはならない。

def _theta_sigma(make_basis, X, y, wt, theta, complex_y):
    """θ の標準誤差を、残差のヤコビアン（有限差分）から出す。 cov = (JᵀJ)⁻¹·s², s²=RSS/(n−p)"""
    def resid(th):
        Phi = make_basis(th, X)
        if Phi is None or not np.all(np.isfinite(Phi)):
            return None
        c = weighted_lstsq(Phi, y, wt)
        r = (Phi @ c - y) * wt
        return np.r_[r.real, r.imag] if complex_y else r
    r0 = resid(np.asarray(theta, float))
    if r0 is None:
        return np.full(len(theta), np.inf)
    n, p = len(r0), len(theta)
    if n <= p:
        return np.full(p, np.inf)
    J = np.zeros((n, p))
    for i in range(p):
        h = 1e-6 * max(abs(theta[i]), 1.0)
        tp = np.asarray(theta, float).copy(); tp[i] += h
        rp = resid(tp)
        if rp is None:
            return np.full(p, np.inf)
        J[:, i] = (rp - r0) / h
    s2 = float(np.sum(r0 ** 2)) / (n - p)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        return np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        return np.full(p, np.inf)


def dyadic_candidates(w: float, max_depth: int = 3):
    """ℤ[1/2] の格子点のうち w に最も近いものを、深さ m の昇順で返す。"""
    best = {}
    for m in range(max_depth + 1):
        v = round(w * (2 ** m)) / (2 ** m)
        d = next(k for k in range(m + 1) if abs(v * (2 ** k) - round(v * (2 ** k))) < 1e-12)
        if v not in best or d < best[v]:
            best[v] = d
    return sorted(best.items(), key=lambda kv: (kv[1], abs(kv[0])))


@dataclass
class SnapReport:
    """指数ひとつ分の判断材料。**丸めは推論でなく物理の知識の行使**なので、
    機械が決められないものは、決めずに **人に見せる**。"""
    idx: int
    w_hat: float
    sigma: float
    candidates: list           # [(値, 深さ m, 何σ離れているか), ...] 深さの昇順
    snapped_to: float | None   # 自動で丸めた先。None なら丸めていない
    k: float                   # 使った閾値

    @property
    def needs_review(self) -> bool:
        return self.snapped_to is None

    def line(self) -> str:
        head = f"θ[{self.idx}] = {self.w_hat:+.6f} ± {self.sigma:.6f}"
        if self.snapped_to is not None:
            ns = abs(self.w_hat - self.snapped_to) / max(self.sigma, 1e-300)
            return f"  {head}  →  **{self.snapped_to:g} に丸め**  ({ns:.1f}σ)"
        out = [f"  {head}  →  **丸めず。目視が要る**"]
        for v, m, ns in self.candidates[:4]:
            out.append(f"      候補 {v:>9g}  (深さ m={m}, 半分 {m} 回)   **{ns:>7.1f}σ** 離れ")
        out.append(f"      → 格子 ℤ[1/2] に乗らない。次のいずれか:")
        out.append(f"        ・次元解析の法則ではない（臨界指数・乱流・生物のアロメトリーなど）")
        out.append(f"        ・格子が足りない（分母に 3 が要る = CFT 系）")
        out.append(f"        ・入力が無次元量で、異常次数が全変数を汚染している")
        out.append(f"        ・データに系統誤差がある（§9.6.1: 測定誤差は指数を 0 の側へ偏らせる）")
        return "\n".join(out)


def snap_set_by_ftest(make_basis, X, y, wt, theta, exponent_idx, n_theta, complex_y,
                      seeds, bounds, rss_free, sigma, max_depth=3, alpha=0.05, verbose=False):
    """丸める集合を **F 検定の後退消去** で決める。

    ① 全部丸めてみる (q=E) → **残りを再学習** → F が通れば採用
    ② 通らなければ、格子から最も遠い指数を外して q=E−1 で再試行
    ③ 何も残らなければ丸めない

      F = [(RSS_snap − RSS_free)/q] / [RSS_free/(n−p)]      F(q, n−p) と比べる

    **利用者の規則**（丸めて再学習して、丸める前と比べて良いかで判断する）。
    誤差棒の検定と漸近的に同じ (F = t²) だが、**線形化を使わないので非線形模型では厳密**。
    実測: 真 w=1.03, V精度 1e-6 で t²=2.3e6 に対し **F=6.2e7**（27倍）。線形化は過小評価する。

    そして **k という旋盤が消える**: 多重度は F の自由度 q が払う（Bonferroni が要らない）。
    「全か無か」も「混合」も、同じ後退消去で扱える:
      純 次元解析 → q=E で通る（全部丸める） ／ 混合 [1,−2,0.63] → q=3 で落ち q=2 で通る
      全部 異常   → q=1 まで落ちて、何も丸めない
    """
    from scipy.stats import f as _f
    n = len(y) * (2 if complex_y else 1)
    p = n_theta + 1
    rss_free = rss_free * n          # **_varpro_core の loss は Σr²/n（平均）。和に直す**
    idx = list(exponent_idx)
    order = sorted(idx, key=lambda i: abs(theta[i] - _nearest_dyadic(theta[i], max_depth))
                                      / max(sigma[i], 1e-300), reverse=True)   # 遠い順（順位付けのみ）
    drop = []
    while len(idx) > len(drop):
        S = [i for i in idx if i not in drop]
        th = np.asarray(theta, float).copy()
        for i in S:
            th[i] = _nearest_dyadic(th[i], max_depth)
        free = np.ones(n_theta, bool)
        for i in S:
            free[i] = False
        if free.any():                                     # 残りを再学習
            th, _ = _varpro_core(make_basis, X, y, wt, lambda r: th, n_theta,
                                 max(8, seeds // 4), complex_y,
                                 free_mask=free, theta_base=th, bounds=bounds)
        Phi = make_basis(th, X)
        c = weighted_lstsq(Phi, y, wt)
        r = (Phi @ c - y) * wt
        rss_snap = float(np.sum(np.abs(r) ** 2))
        q = len(S)
        F = ((rss_snap - rss_free) / q) / max(rss_free / max(n - p, 1), 1e-300)
        crit = float(_f.ppf(1 - alpha, q, max(n - p, 1)))
        if verbose:
            print(f"  [3] F 検定  丸める {q} 個 {[f'θ[{i}]→{th[i]:g}' for i in S]}"
                  f"   F={F:.2f} vs F({q},{n-p}) 5%={crit:.2f}   {'**採用**' if F < crit else '棄却'}")
        if F < crit:
            return S, th
        drop.append(next(i for i in order if i not in drop))
    return [], np.asarray(theta, float).copy()


def _nearest_dyadic(w, max_depth=3):
    q = 2 ** max_depth
    return round(w * q) / q


def k_bonferroni(n_exponents: int, alpha: float = 0.05) -> float:
    """k = Φ⁻¹(1 − α/(2E))。**E = その当てはめで検定する指数の個数**。

    固定の k では取引が解けない (実測): 12法則の力学的E は v⁰ が 2.03σ で偽に落ち (k=2 なら 11/12)、
    μ の嘘 (真 w=1.03) は 2.3σ にある (k≥2.5 なら丸めて −264 と嘘をつく)。0.27σ しか隙間がない。
    **原因は多重検定**: 力学的E は指数を 8個 検定しており、2σ (5%) なら 0.95⁸=0.66 — 3回に1回は偶然落ちる。
    E で補正すれば両方取れる: E=8 → k=2.73 (2.03σ を通す) ／ E=1 → k=1.96 (2.3σ を落とす)。
    """
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / (2 * max(int(n_exponents), 1))))


def snap_dyadic(w: float, sigma: float, max_depth: int = 3, k_sigma: float = 2.0):
    """① 格子 ℤ[1/2] (m ≤ max_depth)  ② |ŵ − p/2^m| < k·σ(ŵ)  ③ m 最小
       通るものが無ければ **None（丸めない）**。複雑なものに逃げない。"""
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    for v, _d in dyadic_candidates(w, max_depth):
        if abs(w - v) < k_sigma * sigma:
            return v
    return None


def fit_separable(make_basis, n_theta: int, X: np.ndarray, y: np.ndarray, *,
                  exponent_idx: Sequence[int] = (),
                  relative: bool = False,
                  seeds: int = 40,
                  init: Callable | None = None,
                  prune_fn: Callable | None = None,
                  snap_depth: int = 3,
                  snap_alpha: float = 0.05,        # F 検定の水準（k という旋盤は無い）
                  complex_y: bool = False,
                  bounds: tuple | None = None,
                  verbose: bool = False) -> SeparableFit:
    """可分モデルを当てはめる。exponent_idx で「次元解析で丸めてよい」成分を宣言する。"""
    wt = 1.0 / np.abs(y) if relative else np.ones(len(y))
    init_fn = init if init else (lambda r: r.normal(0, 1, n_theta))

    # --- 1. VARPRO (全成分) ---
    theta, loss = _varpro_core(make_basis, X, y, wt, init_fn, n_theta, seeds, complex_y,
                               bounds=bounds)
    if verbose: print(f"  [1] VARPRO      theta={np.round(theta,4)}  loss={loss:.3e}")

    # --- 2. 刈り込み (モデル依存) ---
    if prune_fn is not None:
        make_basis, theta, n_theta, exponent_idx = prune_fn(make_basis, theta, X, y, wt,
                                                            n_theta, exponent_idx)
        if verbose: print(f"  [2] 刈り込み後  theta={np.round(theta,4)}")

    # --- 3. 丸め: 格子 ℤ[1/2] (m ≤ snap_depth) × ŵ の誤差棒 (k·σ) ---
    sig = _theta_sigma(make_basis, X, y, wt, theta, complex_y)
    snapped = np.zeros(n_theta, bool)
    report = []
    if len(exponent_idx):
        S, th = snap_set_by_ftest(make_basis, X, y, wt, theta, exponent_idx, n_theta, complex_y,
                                  seeds, bounds, loss, sig, snap_depth, snap_alpha, verbose)
        for i in S:
            snapped[i] = True
    else:
        th = theta.copy()
    for i in exponent_idx:
        cands = [(v, m, abs(theta[i] - v) / max(sig[i], 1e-300))
                 for v, m in dyadic_candidates(theta[i], snap_depth)]
        report.append(SnapReport(i, float(theta[i]), float(sig[i]), cands,
                                 float(th[i]) if snapped[i] else None, float("nan")))
    if verbose and report:
        print(f"  [3] 丸め  (格子 ℤ[1/2] m≤{snap_depth} / F 検定の後退消去, α={snap_alpha})")
        for r in report:
            print(r.line())
        nr = [r for r in report if r.needs_review]
        if nr:
            print(f"  [3] **{len(nr)}/{len(report)} 個は機械が決めない。上の候補と σ を見て、人が決めること。**")

    # --- 4. 再VARPRO (自由な非線形成分のみ。丸めた分は固定) ---
    free = ~snapped
    if snapped.any() and free.any():
        th, loss = _varpro_core(make_basis, X, y, wt,
                                lambda r: th, n_theta, max(8, seeds // 4), complex_y,
                                free_mask=free, theta_base=th, bounds=bounds)
        if verbose: print(f"  [4] 再VARPRO    theta={np.round(th,4)}  loss={loss:.3e}")

    # --- 5. 係数を閉形式で厳密解 ---
    Phi = make_basis(th, X)
    coef = weighted_lstsq(Phi, y, wt)
    predict = lambda Z: make_basis(th, Z) @ coef
    if verbose: print(f"  [5] 係数        {np.round(coef,4)}")
    return SeparableFit(th, coef, snapped, predict, loss, report)


# ------------------------------------------------------------------ 冪和用の刈り込み
def powersum_prune(K: int, F: int):
    """死に項を情報量規準 (BIC) による後退消去で削除する。

    なぜ訓練損失では刈れないか: 死に項は暴走指数で訓練ノイズを局所的に過学習するため、
    消すと訓練損失はむしろ悪化する。損失だけを見る基準では永久に生き残り、
    その指数は同定不能 (c_k≈0 で勾配が消える) なまま外挿の地雷になる。
    BIC は「増やしたパラメータ数に見合う改善か」を問うので、この項を正しく棄却する。
    """
    def _rss(W_sub, X, y, wt):
        if len(W_sub) == 0:
            Phi = np.ones((len(X), 1))
        else:
            Phi = np.c_[np.exp(np.log(X) @ np.asarray(W_sub).reshape(-1, F).T), np.ones(len(X))]
        if not np.all(np.isfinite(Phi)):
            return np.inf
        coef = weighted_lstsq(Phi, y, wt)
        return float(np.sum(np.abs((Phi @ coef - y) * wt) ** 2))

    def _bic(W_sub, X, y, wt):
        rss = _rss(W_sub, X, y, wt)
        if not np.isfinite(rss):
            return np.inf
        n = len(y)
        k = len(W_sub) * F + len(W_sub) + 1        # 指数 + 係数 + bias
        return n * np.log(max(rss / n, 1e-300)) + k * np.log(n)

    def prune(make_basis, theta, X, y, wt, n_theta, exponent_idx):
        W = list(theta.reshape(K, F))
        cur = _bic(W, X, y, wt)
        improved = True
        while improved and len(W) > 1:
            improved = False
            best = None
            for k in range(len(W)):
                trial = W[:k] + W[k+1:]
                b = _bic(trial, X, y, wt)
                if best is None or b < best[0]:
                    best = (b, trial)
            if best is not None and best[0] < cur:     # 消した方が BIC が良い → 死に項
                cur, W = best[0], best[1]
                improved = True
        K2 = len(W)
        new_basis = lambda th, Z: np.c_[np.exp(np.log(Z) @ th.reshape(K2, F).T), np.ones(len(Z))]
        return new_basis, np.asarray(W).ravel(), K2 * F, tuple(range(K2 * F))
    return prune


# ================================================================== デモ
if __name__ == "__main__":
    def mse(f, X, t): return float(np.mean(np.abs(f(X) - t(X)) ** 2))
    rng = np.random.default_rng(20260716)

    print("=" * 74)
    print("デモA: 混在モデル  y = 2.5·t²·exp(−0.37t)")
    print("  n=2 は指数型 → 丸める / γ=0.37 は経験的 → 丸めない / 両者は相関する")
    print("=" * 74)
    truth = lambda X: 2.5 * X[:, 0] ** 2 * np.exp(-0.37 * X[:, 0])
    Xtr = rng.uniform(0.1, 5, (300, 1)); Xex = rng.uniform(5, 9, (600, 1))
    y = truth(Xtr) + rng.normal(0, 0.01 * np.std(truth(Xtr)), 300)

    def basis(th, Z):                      # th = [n, log gamma];  A は線形
        n, g = th[0], np.exp(th[1])
        t = Z[:, 0]
        if n > 12 or g > 30: return None
        return np.c_[t ** n * np.exp(-g * t), np.ones(len(Z))]
    init = lambda r: np.array([r.uniform(0, 5), r.normal(0, 1)])

    # (a) 丸めなし
    fa = fit_separable(basis, 2, Xtr, y, init=init, exponent_idx=())
    print(f"{'(a) 丸めなし':34s} n={fa.theta[0]:.4f} γ={np.exp(fa.theta[1]):.5f} A={fa.coef[0]:.4f}"
          f"  外挿MSE {mse(fa.predict,Xex,truth):.4e}")
    # (b) 丸めるが再学習しない = (a)の解を丸めて、係数だけ解き直す
    th = fa.theta.copy(); th[0] = round(th[0] * 2) / 2
    coef_b = weighted_lstsq(basis(th, Xtr), y, np.ones(len(y)))
    pred_b = lambda Z: basis(th, Z) @ coef_b
    print(f"{'(b) 丸めるが γ を再学習しない':34s} n={th[0]:.4f} γ={np.exp(th[1]):.5f} A={coef_b[0]:.4f}"
          f"  外挿MSE {mse(pred_b,Xex,truth):.4e}")
    # (c) 一般形: 丸め→固定→残りの非線形(γ)を再学習→係数を閉形式
    fc = fit_separable(basis, 2, Xtr, y, init=init, exponent_idx=(0,))
    print(f"{'(c) ★一般形: 丸め→固定→γ を再学習':34s} n={fc.theta[0]:.4f} γ={np.exp(fc.theta[1]):.5f}"
          f" A={fc.coef[0]:.4f}  外挿MSE {mse(fc.predict,Xex,truth):.4e}")
    print(f"{'真値':34s} n=2      γ=0.37    A=2.5")

    print("\n" + "=" * 74)
    print("デモB: 一般形で元の4課題 (冪和は全指数が丸まる → 再VARPRO は自動的に不要)")
    print("=" * 74)
    tg = lambda X: 2.7 * X[:, 0] * X[:, 1] * X[:, 2] ** -2
    Gtr = np.c_[rng.uniform(1, 10, 300), rng.uniform(1, 10, 300), rng.uniform(1, 5, 300)]
    Gex = np.c_[rng.uniform(0.3, 20, 900), rng.uniform(0.3, 20, 900), rng.uniform(0.3, 10, 900)]
    yg = tg(Gtr) * np.exp(rng.normal(0, 0.01, 300))
    bg = lambda th, Z: np.c_[np.exp(np.log(Z) @ th.reshape(2, 3).T), np.ones(len(Z))]
    EXP_BOUND = 6.0            # 次元解析: 物理の冪が |w|>6 になることはまずない
    fit = fit_separable(bg, 6, Gtr, yg, exponent_idx=tuple(range(6)), relative=True,
                        prune_fn=powersum_prune(2, 3),
                        bounds=(-EXP_BOUND*np.ones(6), EXP_BOUND*np.ones(6)), verbose=True)
    print(f"  gravitation 外挿MSE {mse(fit.predict, Gex, tg):.4e}   "
          f"指数{np.round(fit.theta,3)} 係数{fit.coef[0]:.4f}  (真 2.7, [1,1,-2])")


# ================================================================ 冪和の高水準 API
# 今日の一本の線（§7.5.4）:
#   項の数        → **離散**（BIC 前進選択）      §7
#   項の中の変数    → **離散**（BIC 前進選択）      §7.5   ← w=0 は「w の値」でなく「変数が居ない」
#   指数の値       → 連続（VarPro + LM）         §5
#   格子への丸め    → **離散**（F 検定 → 人）      §9.8
# 離散なものを連続に見せると、崖ができる（0⁰ の 1,875倍の井戸）か、嘘になる（丸めの 3.4%）。

def power_unit(x, w):
    """冪ユニットの **全域化された** 演算（§7.5.7）。

        x^0 = 1   全ての x で        **空の積。譲れない**（v⁰ が要る。0⁰=0 にすると
                                     静止した物体の位置エネルギーが消える）
        0^w = 0   w > 0             正しい（E=½mv² は v=0 で E=0）
        **0^w = 0   w < 0**          **真値は ∞。嘘。だが有界で、検出可能で、当てはめを救う**

    最後の一行が要点である。実測（r^{−2} の r が測定器の分解能以下で 0 と記録された点。`zero_total.py`）:

        規約 ∞（正直）: 1% の点で **w(r) = 0.0000**。`w<0` を試すと inf が出るので
                        **負の指数を一切使えなくなり、境界に貼り付く。逆二乗が消える**
        **規約 0**    : **15% 壊れていても w(r) = −2.0004**。頑健な損失すら要らない

    機構: 模型が 0 を返し y が巨大なので相対残差は (0−y)/y = **−1 で頭打ち**。
          壊れた点は 1点あたり最大 1 しか損失に寄与できず、良い点が支配権を保つ。
          そして残差 1.000 で完全に分離する（良い点の 150倍）ので、
          **「この点はこの法則で説明できない」が読める**。

    > **有界な嘘は、正直な無限大に勝つ。**（Lean/mathlib が `x/0 := 0` とするのと同じ論法:
    >  演算を全域にして、定理には仮定を付ける。仮定が破れた所では junk が返るが、
    >  **有限に返るので機械は回り続ける**。）
    """
    x = np.asarray(x, float)
    if w == 0.0:
        return np.ones_like(x)
    with np.errstate(all="ignore"):
        v = np.power(np.abs(x), w)
    return np.where(np.abs(x) == 0.0, 0.0, v)          # w>0 は正しく 0、w<0 は有界な嘘


def _ps_basis(masks, Ws, X):
    """項ごとに「居る変数」だけの積。**居ない変数は評価しない**（0⁰ が現れない）。"""
    cols = []
    for S, w in zip(masks, Ws):
        c = np.ones(len(X))
        for j, wj in zip(S, w):
            c = c * power_unit(X[:, j], wj)
        cols.append(c)
    return np.c_[tuple(cols) + (np.ones(len(X)),)] if cols else np.ones((len(X), 1))


def _ps_fit(masks, X, y, wt, seeds=20, bound=6.0):
    npar = sum(len(S) for S in masks)
    if npar == 0:
        Phi = _ps_basis(masks, [], X)
        return [], float(np.sum(np.abs((Phi @ weighted_lstsq(Phi, y, wt) - y) * wt) ** 2))
    def unpack(v):
        out, i = [], 0
        for S in masks:
            out.append(v[i:i + len(S)]); i += len(S)
        return out
    def resid(v):
        Phi = _ps_basis(masks, unpack(v), X)
        if not np.all(np.isfinite(Phi)):
            return np.full(len(y), 1e6)
        return (Phi @ weighted_lstsq(Phi, y, wt) - y) * wt
    lo, hi = -bound * np.ones(npar), bound * np.ones(npar)
    best = None
    for s in range(seeds):
        v0 = np.clip(np.random.default_rng(s + 7).normal(0, 1, npar), lo + 1e-9, hi - 1e-9)
        try:
            o = least_squares(resid, v0, max_nfev=4000, bounds=(lo, hi))
            val = float(np.sum(o.fun ** 2))
            if best is None or val < best[0]:
                best = (val, o.x)
        except Exception:
            pass
    return (unpack(best[1]), best[0]) if best else ([np.zeros(len(S)) for S in masks], np.inf)


def _ps_bic(masks, X, y, wt, seeds=20):
    Ws, rss = _ps_fit(masks, X, y, wt, seeds)
    n = len(y); k = sum(len(S) for S in masks) + len(masks) + 1
    return n * np.log(max(rss / n, 1e-300)) + k * np.log(n), Ws, rss


@dataclass
class PowerSumFit:
    masks: list                # 項ごとの「居る変数」の添字
    Ws: list                   # 項ごとの指数（居る変数の分だけ）
    coef: np.ndarray           # 係数（末尾は bias）
    snapped: list              # 項ごとの bool
    predict: Callable
    rss: float
    report: list
    relative: bool

    def formula(self, names=None) -> str:
        names = names or [f"x{j}" for j in range(64)]
        ts = []
        for S, w, c in zip(self.masks, self.Ws, self.coef[:len(self.masks)]):
            body = "·".join(f"{names[j]}^{wj:g}" for j, wj in zip(S, w))
            ts.append(f"{c:+.4g}·{body}" if body else f"{c:+.4g}")
        if abs(self.coef[-1]) > 0:
            ts.append(f"{self.coef[-1]:+.3g}")
        return " + ".join(ts) if ts else "0"


def fit_powersum(X, y, *, Kmax=4, relative=None, seeds=25, snap_depth=3, snap_alpha=0.05,
                 bound=6.0, verbose=False) -> PowerSumFit:
    """冪和 `Σₖ cₖ ∏_{j∈Sₖ} xⱼ^{wₖⱼ}` を当てる。構造は離散に、指数は連続に。

    ① **重み**: `relative=None` なら `y` に 0 があるかで自動選択。
       **乗法ノイズは 0 を 0 のままにする**（`0 × e^ε = 0`）ので、`y=0` が観測されうるなら
       そこのノイズは加法である。→ `y` に 0 があれば絶対重み（§7.5.2）。
    ② **構造**: 項と、項の中の変数を **BIC 前進選択**。
       **`w=0` は「`w` の値のひとつ」ではなく「その変数が居ない」**（§7.5.3）。
       居ない変数は基底に入らないので **`0⁰` を一度も評価しない ⟹ 崖が無い**。
    ③ **指数**: VarPro（係数は閉形式）+ Levenberg–Marquardt + 多点初期化。
    ④ **丸め**: 格子 `ℤ[1/2]` × **F 検定の後退消去**（§9.8.5）。決められないものは `report` で返す。
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    F = X.shape[1]
    # 重みの自動選択は、**ノイズモデルの判別**である（§7.5.2）:
    #   **乗法ノイズは y の符号を変えられない**（y × e^ε は符号がそのまま。0 は 0 のまま）
    #   **0 のまわりの加法ノイズは、符号を変える**（0 + N(0,σ) は両符号を取る）
    # ⟹ y が **両符号を取る** か **厳密な 0 を含む** なら、そこのノイズは加法であり、
    #    相対重み 1/|y| は爆発する。それ以外は相対（乗法ノイズの最尤）。
    #
    # **「動的範囲が広い」は根拠にならない**（オームの法則 V=IR は y が 0.1〜500 を動くが、
    #  ノイズは乗法である）。実測: min|y|/median|y| < 1e-2 で切り替えると 12/12 → 9/12 に落ちた。
    ay = np.abs(y)
    has_exact_zero = bool(np.any(ay < 1e-300))
    both_signs = bool(np.any(y > 0) and np.any(y < 0))
    additive = has_exact_zero or both_signs
    if relative is None:
        relative = not additive
    wt = 1.0 / np.maximum(ay, 1e-300) if relative else np.ones(len(y))
    if verbose:
        why = ("y が厳密な 0 を含む" if has_exact_zero else
               ("y が **両符号** を取る → 乗法ノイズではありえない" if both_signs else "y の符号は一定"))
        print(f"  [0] 重み  {'相対 1/|y|' if relative else '**絶対 1**'}   ({why})")

    # --- ①② 構造: 項と変数を BIC 前進選択 ---
    masks = []
    cur, Ws, _ = _ps_bic(masks, X, y, wt, seeds)
    while True:
        best = None
        cands = [(k, j) for k in range(len(masks)) for j in range(F) if j not in masks[k]]
        if len(masks) < Kmax:
            cands += [(len(masks), j) for j in range(F)]          # 新しい項を開く
        for k, j in cands:
            cand = [tuple(m) for m in masks]
            if k == len(masks):
                cand.append((j,))
            else:
                cand[k] = tuple(sorted(masks[k] + (j,)))
            b, W_, _ = _ps_bic(cand, X, y, wt, seeds)
            if best is None or b < best[0]:
                best = (b, cand, W_, k, j)
        if best is None or best[0] >= cur:
            break
        cur, masks, Ws = best[0], best[1], best[2]
        if verbose:
            print(f"  [2] 構造  + 項{best[3]+1} に x{best[4]}   BIC → {cur:.1f}")

    # --- ④ 丸め: 格子 ℤ[1/2] × F 検定の後退消去 ---
    Ws, rss_free = _ps_fit(masks, X, y, wt, seeds, bound)
    flat = [(k, i) for k, S in enumerate(masks) for i in range(len(S))]
    n = len(y); p = len(flat) + len(masks) + 1
    snapped = [np.zeros(len(S), bool) for S in masks]
    if flat:
        sig = _ps_sigma(masks, Ws, X, y, wt)
        order = sorted(flat, key=lambda t: abs(Ws[t[0]][t[1]] - _nearest_dyadic(Ws[t[0]][t[1]], snap_depth))
                                           / max(sig[t[0]][t[1]], 1e-300), reverse=True)
        drop = []
        while len(flat) > len(drop):
            S_ = [t for t in flat if t not in drop]
            Wn = [w.copy() for w in Ws]
            for k, i in S_:
                Wn[k][i] = _nearest_dyadic(Ws[k][i], snap_depth)
            free = [t for t in flat if t in drop]
            if free:                                              # 残りの指数を再学習
                def r(v):
                    Wt = [w.copy() for w in Wn]
                    for (k, i), val in zip(free, v): Wt[k][i] = val
                    Phi = _ps_basis(masks, Wt, X)
                    if not np.all(np.isfinite(Phi)): return np.full(n, 1e6)
                    return (Phi @ weighted_lstsq(Phi, y, wt) - y) * wt
                try:
                    o = least_squares(r, [Ws[k][i] for k, i in free], max_nfev=3000,
                                      bounds=(-bound*np.ones(len(free)), bound*np.ones(len(free))))
                    for (k, i), val in zip(free, o.x): Wn[k][i] = val
                except Exception:
                    pass
            Phi = _ps_basis(masks, Wn, X)
            rss_s = float(np.sum(np.abs((Phi @ weighted_lstsq(Phi, y, wt) - y) * wt) ** 2))
            q = len(S_)
            Fv = ((rss_s - rss_free) / q) / max(rss_free / max(n - p, 1), 1e-300)
            crit = float(_fdist.ppf(1 - snap_alpha, q, max(n - p, 1)))
            if verbose:
                print(f"  [4] 丸め  {q} 個   F={Fv:.2f} vs F({q},{n-p}) {snap_alpha:.0%}={crit:.2f}"
                      f"   {'**採用**' if Fv < crit else '棄却'}")
            if Fv < crit:
                Ws = Wn
                for k, i in S_: snapped[k][i] = True
                break
            drop.append(next(t for t in order if t not in drop))

    sig = _ps_sigma(masks, Ws, X, y, wt)
    report = []
    for k, S in enumerate(masks):
        for i, j in enumerate(S):
            cands = [(v, m, abs(Ws[k][i] - v) / max(sig[k][i], 1e-300))
                     for v, m in dyadic_candidates(Ws[k][i], snap_depth)]
            report.append(SnapReport(j, float(Ws[k][i]), float(sig[k][i]), cands,
                                     float(Ws[k][i]) if snapped[k][i] else None, float("nan")))
    Phi = _ps_basis(masks, Ws, X)
    coef = weighted_lstsq(Phi, y, wt)
    rss = float(np.sum(np.abs((Phi @ coef - y) * wt) ** 2))
    if verbose:
        for r_ in report:
            print(r_.line())
        nr = [r_ for r_ in report if r_.needs_review]
        if nr:
            print(f"  [4] **{len(nr)}/{len(report)} 個は機械が決めない。人が決めること。**")
    return PowerSumFit(masks, Ws, coef, snapped,
                       lambda Z: _ps_basis(masks, Ws, np.asarray(Z, float)) @ coef,
                       rss, report, relative)


def _ps_sigma(masks, Ws, X, y, wt):
    """指数の標準誤差を、残差のヤコビアン（有限差分）から。"""
    flat = [(k, i) for k, S in enumerate(masks) for i in range(len(S))]
    out = [np.full(len(S), np.inf) for S in masks]
    if not flat: return out
    def r(Wx):
        Phi = _ps_basis(masks, Wx, X)
        if not np.all(np.isfinite(Phi)): return None
        return (Phi @ weighted_lstsq(Phi, y, wt) - y) * wt
    r0 = r(Ws)
    if r0 is None: return out
    n, pn = len(r0), len(flat)
    if n <= pn: return out
    J = np.zeros((n, pn))
    for c, (k, i) in enumerate(flat):
        h = 1e-6 * max(abs(Ws[k][i]), 1.0)
        Wp = [w.copy() for w in Ws]; Wp[k][i] += h
        rp = r(Wp)
        if rp is None: return out
        J[:, c] = (rp - r0) / h
    s2 = float(np.sum(r0 ** 2)) / (n - pn)
    try:
        d = np.sqrt(np.maximum(np.diag(np.linalg.inv(J.T @ J) * s2), 0.0))
    except np.linalg.LinAlgError:
        return out
    for c, (k, i) in enumerate(flat): out[k][i] = d[c]
    return out
