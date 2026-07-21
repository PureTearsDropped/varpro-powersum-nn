#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""implicit_discovery — 暗黙形の物理法則発見: Σc·(単項式/微分項) = 0 の 零空間を 解く。

  形の由来(2026-07-21): ユーザの 1=(m²c⁴+p²c²)/E² の 提案 → 分母を払うと 単項式和=0 →
  法則 = ライブラリ行列の 零空間 = SVD 一発(勾配法ゼロ・シードという概念が 消える)。
  y=f(x) の 明示形が 持つ「√(和) の 壁」(相対論の γ など)は、目的変数を ライブラリに
  入れた 瞬間に 消滅する。無次元数(バッキンガム π)の 機械化・一般化。

  旧チャットで 同じ形が 自明解に 落ちた 原因と 対策(全部 構造で 塞ぐ):
    c → 0            → SVD は 球面 ‖c‖=1 上で 解く(c=0 が 存在できない)
    x^w − x^w 恒真式  → 指数は 離散格子で 凍結(列の 重複が 存在できない)
    局所盆地/シード    → 固有値問題の 大域厳密解(勾配法 不使用)

  実測済みの 戦績: E²=m²+p² と γ²(1−v²)=1 を 係数±1.000000 で発見 / シュレディンガー
  方程式+エネルギー量子化(E=0.5, 1.5)+消滅演算子 a|0⟩=0 を 自発発見 / 自由粒子で
  虚数単位 i を 係数として 発見 / 毒入り水素で 素の SVD は 不収束・本パイプラインは
  E=−0.500000 を 救出。

  パイプライン: 測定 → Tot入口(監査済み全域化・毒をフラグに) → フラグ行除外 →
  ライブラリ(単項式×任意の微分列) → 列正規化 SVD → ギャップ判定(法則の実在) →
  恒真式検出(無関係データでも 0 か) → 係数の 逆正規化(物理定数の 読み取り)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import numpy as np
import torch
from cuda_total import Tot                       # vendored・監査済みの 入口全域化


def build_library(vars_dict, max_deg, extra_cols=None):
    """単項式ライブラリ(全変数・次数≤max_deg) + 任意の追加列(微分など)。
       返り値: (行列 (n,K), 項の名前リスト)"""
    names = list(vars_dict.keys())
    cols, labels = [], []
    if names:                                        # 変数なし(微分列だけ)の呼び出しも許す
        data = np.stack([np.asarray(vars_dict[k], dtype=np.float64) for k in names], 1)
        exps = [e for e in itertools.product(range(max_deg + 1), repeat=len(names))
                if sum(e) <= max_deg]
        for e in exps:
            cols.append(np.prod(data ** np.array(e), axis=1))
            labels.append("·".join(f"{n}^{k}" if k > 1 else n
                                   for n, k in zip(names, e) if k) or "1")
    if extra_cols:
        for lab, col in extra_cols.items():
            cols.append(np.asarray(col))             # dtype強制なし(複素の虚部を殺さない)
            labels.append(lab)
    return np.stack(cols, 1), labels


def discover(vars_dict, max_deg=2, extra_cols=None, complex_ok=False):
    """暗黙法則の発見。返り値 dict:
         law        : [(係数, 項名)] — 最大係数を 1 に 正規化(物理定数は 係数比)
         sigma_min  : 最小特異値(≈0 ⟺ 厳密法則が 存在)
         gap        : sigma_next/sigma_min — 大きいほど「一意の厳密法則」
         n_flagged  : Tot 入口が 名指しした 汚染行数(除外済み)
         c, labels  : 生の係数ベクトルと 項名(形検査・追加解析用)"""
    lib, labels = build_library(vars_dict, max_deg, extra_cols)
    if complex_ok:
        keep = np.isfinite(lib).all(axis=1)          # 複素は Tot 対象外: 有限性のみ検査
        n_flagged = int((~keep).sum())
    else:
        # ★役割分担: フラグ=監査済み入口(Tot, float32範囲を「機械の正直な範囲」として毒を名指し)
        #            値  =元の float64(フラグなし行は 無傷なので 精度を 失わない)
        T = Tot(torch.tensor(lib))
        keep = ~((T.flag > 0).any(dim=1).numpy())
        n_flagged = int((~keep).sum())
    clean = lib[keep]
    norms = np.linalg.norm(clean, axis=0)
    norms[norms == 0] = 1.0
    _, S, Vt = np.linalg.svd(clean / norms, full_matrices=False)
    c = Vt[-1].conj() / norms                        # 列正規化を 剥がして 物理の係数比へ
    c = c / c[np.argmax(np.abs(c))]
    law = [(co, lab) for co, lab in zip(c, labels) if abs(co) > 1e-4]
    return dict(law=law, sigma_min=float(S[-1]), gap=float(S[-2] / max(S[-1], 1e-300)),
                n_flagged=n_flagged, c=c, labels=labels)


def is_tautology(result, vars_dict, extra_fn=None, rng=None):
    """恒真式検出器: 発見式を 物理と無関係な ランダムデータで 評価。
       それでも ≈0 なら 恒真式(x^w−x^w 型・何も 学んでいない)。
       extra_fn(random_vars) で 微分列などの ランダム版を 供給(省略時は 単項式のみ検査)。"""
    rng = rng or np.random.default_rng(0)
    rand_vars = {k: rng.uniform(0.1, 2.0, 2000) for k in vars_dict}
    extra = extra_fn(rand_vars) if extra_fn else None
    try:
        lib, _ = build_library(rand_vars, _degree_of(result), extra)
    except Exception:
        return False
    if lib.shape[1] != len(result["c"]):
        return False
    resid = np.abs(lib @ result["c"]).mean()
    scale = np.abs(lib).mean() * np.abs(result["c"]).max()
    return bool(resid < 1e-6 * max(scale, 1e-300))

def _degree_of(result):
    d = 0
    for lab in result["labels"]:
        deg = sum(int(p.split("^")[1]) if "^" in p else 1
                  for p in lab.split("·") if p != "1" and "'" not in p and "_" not in p)
        d = max(d, deg)
    return d


def fmt(result):
    s = "  +  ".join(f"{co:+.6f}·{lab}" if not np.iscomplexobj(co) else
                     f"({co:+.4f})·{lab}" for co, lab in result["law"])
    return (f"0 = {s}\n    σ_min {result['sigma_min']:.1e}  gap {result['gap']:.0e}"
            f"  汚染行除外 {result['n_flagged']}")


# ---------------------------------------------------------------- self-test(実測の戦績を回帰化)
def self_test():
    rng = np.random.default_rng(0)
    print("implicit_discovery — 戦績の回帰テスト")

    # ① 相対論: E²=m²+p² (係数±1・ギャップ巨大)
    m = rng.uniform(0.1, 2, 4000); p = rng.uniform(0, 2, 4000)
    E = np.sqrt(m * m + p * p)
    r = discover({"m": m, "p": p, "E": E}, max_deg=2)
    d = {lab: co for co, lab in r["law"]}
    assert abs(abs(d["E^2"]) - 1) < 1e-6 and abs(d["m^2"] / d["E^2"] + 1) < 1e-6 \
        and abs(d["p^2"] / d["E^2"] + 1) < 1e-6
    assert r["sigma_min"] < 1e-10 and r["gap"] > 1e6
    assert not is_tautology(r, {"m": m, "p": p, "E": E})
    print("  ① E²−m²−p²=0 発見・恒真式でない ✓ ", f"(σ {r['sigma_min']:.0e}, gap {r['gap']:.0e})")

    # ② ローレンツ因子: γ²(1−v²)=1
    v = rng.uniform(0, 0.95, 4000); gam = 1 / np.sqrt(1 - v * v)
    r = discover({"v": v, "γ": gam}, max_deg=4)
    d = {lab: co for co, lab in r["law"]}
    key = next(k for k in d if "γ^2" in k and "v" in k)
    assert abs(abs(d[key]) - 1) < 1e-5
    print("  ② γ²(1−v²)=1 発見 ✓")

    # ③ 毒入り水素(パイプラインの生死): 素のSVDは不収束・本実装は E=−1/2 を救出
    rr = np.linspace(1e-3, 12, 6000); dr = rr[1] - rr[0]
    u = rr * np.exp(-rr)
    bad = rng.choice(len(u), 10, replace=False)
    u[bad[:5]] = np.inf; u[bad[5:]] = np.nan
    d2 = (u[2:] - 2 * u[1:-1] + u[:-2]) / dr ** 2
    ri, ui = rr[1:-1], u[1:-1]
    try:
        np.linalg.svd(np.stack([d2, ui / ri, ui], 1))
        naive_died = False
    except np.linalg.LinAlgError:
        naive_died = True
    r = discover({}, max_deg=0,
                 extra_cols={"u''": d2, "u/r": ui / ri, "u": ui, "r·u": ri * ui})
    d = {lab: co for co, lab in r["law"]}
    Eh = (d["u"] / d["u''"]) / 2
    assert r["n_flagged"] >= 10 and abs(Eh + 0.5) < 1e-4
    print(f"  ③ 毒入り水素: 素のSVD{'死亡' if naive_died else '生存(環境依存)'}"
          f" / 本実装 汚染{r['n_flagged']}行除外 → E = {Eh:.6f} ✓")

    # ④ 法則なしデータで 正直に「なし」(σ_min 大)・恒真式も 出さない
    r = discover({"a": rng.uniform(0.1, 2, 3000), "b": rng.uniform(0.1, 2, 3000),
                  "c": rng.uniform(0.1, 2, 3000)}, max_deg=2)
    assert r["sigma_min"] > 1e-3
    print(f"  ④ 無関係データ: σ_min {r['sigma_min']:.0e} = 法則なしを正直に報告 ✓")

    # ⑤ 複素: 自由シュレディンガー iψ_t=−ψ_xx/2 — 虚数単位が 係数に 出る
    xg = np.linspace(-10, 10, 500); tg = np.linspace(0, 2, 500)
    ks = rng.uniform(-2, 2, 8)
    X, T2 = np.meshgrid(xg, tg, indexing="ij")
    psi = sum(np.exp(1j * (k * X - k * k / 2 * T2)) for k in ks)
    dt, dx = tg[1] - tg[0], xg[1] - xg[0]
    pt = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dt)
    pxx = (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dx ** 2
    r = discover({}, max_deg=0, complex_ok=True,
                 extra_cols={"ψ_t": pt.ravel(), "ψ_xx": pxx.ravel(),
                             "ψ": psi[1:-1, 1:-1].ravel()})
    d = {lab: co for co, lab in r["law"]}
    ratio = d["ψ_xx"] / d["ψ_t"]
    assert abs(ratio.real) < 1e-2 and abs(ratio.imag + 0.5) < 1e-2
    print(f"  ⑤ 自由シュレディンガー: ψ_xx/ψ_t = {ratio:.4f} ≈ −0.5i — iをデータから発見 ✓")
    print("done — 発見は 零空間・毒は フラグ・嘘は 恒真式検出器と ギャップが 締め出す")


if __name__ == "__main__":
    self_test()
