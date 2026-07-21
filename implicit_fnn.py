#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""implicit_fnn — 「左辺 = 0」を FNN の 先生にする (implicit_discovery × NN 学習)。

  implicit_discovery は データから 暗黙法則 Θ·c = 0 を 零空間で 発見する 装置。
  本実験は その 左辺を **損失** に 変える — VarPro の 分離を 暗黙法則に:
  内側 (c) は 零空間で 厳密に 解き、外側 (ネット) は 残差 |Θ·c|² を 下げる。

  Phase A — 法則だけが 先生 (教師データ 0 点):
      f(x+y) = f(x)·f(y)  (+ 錨 f(1)=e)   → MLP は exp に なるか
      f(x+y)+f(x−y) = 2f(x)f(y)  (+ f(1)=cos1) → cos に なるか
      さらに「法則の橋」: 訓練域外の f(z) を 法則で 域内に 還元して 評価
      (exp: f(z)=f(z/2ᵏ)^(2ᵏ) / cos: 倍角 f(2x)=2f(x)²−1) vs 素の 外挿。

  Phase B — 発見した 法則を 無ラベルの 先生に (半教師):
      E=√(m²+p²)。N 点の ラベルから c を 零空間発見 → 固定し、
      無ラベル点で |Θ̃(m,p,ĝ(m,p))·c|² を 追加損失に。
      腕: ①基準(ラベルのみ) ②発見則 ③真値則(天井)。8 シード・中央値と 最悪。

  Phase C — 汚染 + フラグ (全域算術の 出番):
      無ラベル集合に Inf/NaN と 巨大値を 注入。IEEE 腕は 素通し、
      フラグ腕は 入口/ライブラリ段の 非有限で 行を 名指し 除外。生存率を 測る。

  規律: 8 シード + 最悪ケース報告・負けも そのまま 印字 (measured, never assumed)。
"""
import math
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def mlp(inp=1, width=64):
    return nn.Sequential(nn.Linear(inp, width), nn.Tanh(),
                         nn.Linear(width, width), nn.Tanh(),
                         nn.Linear(width, 1)).to(DEV)


def train(loss_fn, net, steps=4000, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        l = loss_fn(net)
        l.backward()
        opt.step()
    return net


# ================================================================ Phase A
def phase_a():
    print("Phase A — 法則だけが 先生 (教師データ 0 点・錨 1 点)")
    rng = torch.Generator(device=DEV).manual_seed(0)

    # ---- exp: f(x+y) = f(x)f(y)
    def loss_exp(net):
        x = 0.5 * torch.rand(4096, 1, generator=rng, device=DEV)
        y = 0.5 * torch.rand(4096, 1, generator=rng, device=DEV)
        law = (net(x + y) - net(x) * net(y)).pow(2).mean()
        anchor = (net(torch.ones(1, 1, device=DEV)) - math.e).pow(2).sum()
        return law + anchor
    f = train(loss_exp, mlp(), steps=8000)
    z = torch.linspace(0, 1, 201, device=DEV).reshape(-1, 1)
    err = (f(z).squeeze() - torch.exp(z).squeeze()).abs().max().item()
    print(f"  exp: 加法法則+錨(f(1)=e) → sup|f−exp| on [0,1] = {err:.1e}")

    # ---- cos: f(x+y)+f(x−y) = 2f(x)f(y)  (d'Alembert — cos と cosh の 2 枝を 持つ)
    # 錨が 正値 1 点だけだと ネットは cosh 枝へ 逃げる (実測: sup誤差 1.0 = cosh(1)−cos(1))。
    # 負値の 錨 f(2)=cos2<0 が 枝を 選ぶ (cosh は 正しか 取れない)。
    anc_x = torch.tensor([[1.0], [2.0]], device=DEV)
    anc_y = torch.tensor([[math.cos(1.0)], [math.cos(2.0)]], device=DEV)
    def loss_cos(net):
        x = torch.rand(4096, 1, generator=rng, device=DEV)
        y = torch.rand(4096, 1, generator=rng, device=DEV)
        law = (net(x + y) + net(x - y) - 2 * net(x) * net(y)).pow(2).mean()
        anchor = (net(anc_x) - anc_y).pow(2).sum()
        return law + 10.0 * anchor
    g = train(loss_cos, mlp(), steps=8000)
    errc = (g(z).squeeze() - torch.cos(z).squeeze()).abs().max().item()
    print(f"  cos: d'Alembert則+負値錨(f(2)=cos2) → sup|f−cos| on [0,1] = {errc:.1e}"
          f"  (正値錨のみだと cosh 枝へ 逃げる — 枝選択は 錨の 仕事)")

    # ---- 法則の橋: 域外 z∈{1.5, 2, 3} を 法則で 域内に 還元
    print("  法則の橋 (訓練域 [0,1] の 外):")
    print(f"    {'z':>4} {'真値 e^z':>10} {'素の外挿':>10} {'法則還元':>10}")
    for zv in (1.5, 2.0, 3.0):
        k = math.ceil(math.log2(zv))                     # z/2^k ≤ 1
        base = f(torch.tensor([[zv / 2**k]], device=DEV)).item()
        bridged = base ** (2 ** k)
        naive = f(torch.tensor([[zv]], device=DEV)).item()
        true = math.exp(zv)
        print(f"    {zv:>4} {true:>10.3f} {naive:>10.3f} {bridged:>10.3f}"
              f"   (誤差 {abs(naive-true):8.3f} → {abs(bridged-true):8.3f})")


# ================================================================ Phase B / C 共通
def make_data(seed, n_lab, n_unlab=2000, noise=0.01):
    rs = np.random.default_rng(seed)
    def sample(n):
        m = rs.uniform(0.5, 2.0, n)
        p = rs.uniform(0.5, 2.0, n)
        return m, p, np.sqrt(m * m + p * p)
    ml, pl, El = sample(n_lab)
    El = El + noise * rs.standard_normal(n_lab)
    mu, pu, _ = sample(n_unlab)
    mt, pt, Et = sample(4000)
    return (ml, pl, El), (mu, pu), (mt, pt, Et)


COLS = ["1", "m", "p", "E", "m2", "p2", "E2"]

def library(m, p, E):
    "Θ の 7 列 (真の 零ベクトル = E²−m²−p²)。torch/numpy 両対応。"
    one = torch.ones_like(E) if torch.is_tensor(E) else np.ones_like(E)
    Z = [one, m, p, E, m * m, p * p, E * E]
    return torch.stack(Z, -1) if torch.is_tensor(E) else np.stack(Z, -1)


def discover_c(ml, pl, El):
    "ラベルから 法則を 零空間発見 (implicit_discovery.discover と 同じ 芯・列RMS正規化)。"
    Th = library(ml, pl, El)
    s = np.sqrt((Th ** 2).mean(0)) + 1e-30
    _, sv, Vt = np.linalg.svd(Th / s, full_matrices=False)
    c = (Vt[-1] / s)
    gap = sv[-2] / max(sv[-1], 1e-30)
    return c / np.abs(c).max(), gap


C_TRUE = np.array([0, 0, 0, 0, -1.0, -1.0, 1.0])         # E²−m²−p²


def fit(seed, n_lab, arm, contaminate=False, honesty="ieee", lam=1.0):
    (ml, pl, El), (mu, pu), (mt, pt, Et) = make_data(seed, n_lab)
    if contaminate:                                       # 5% Inf/NaN + 2.5% 巨大値
        rs = np.random.default_rng(seed + 999)
        idx = rs.choice(len(mu), len(mu) // 20, replace=False)
        mu[idx[::2]] = np.inf
        pu[idx[1::2]] = np.nan
        idx2 = rs.choice(len(mu), len(mu) // 40, replace=False)
        mu[idx2] = 1e200                                  # 入口は 有限・E² で 爆発する 種
    if arm == "discovered":
        c, _ = discover_c(ml, pl, El)
    elif arm == "oracle":
        c = C_TRUE.copy()
    else:
        c = None

    torch.manual_seed(seed)
    net = mlp(inp=2)
    tl = torch.tensor(np.stack([ml, pl], 1), device=DEV)
    yl = torch.tensor(El, device=DEV).reshape(-1, 1)
    tu = torch.tensor(np.stack([mu, pu], 1), device=DEV)
    ct = None if c is None else torch.tensor(c, device=DEV)

    keep = None
    if c is not None and honesty == "flags":
        # 入口の 税関: 非有限を 名指し (全域算術の 入口 全域化と 同じ 役)
        keep = torch.isfinite(tu).all(1)

    def loss(net):
        l = (net(tl) - yl).pow(2).mean()
        if ct is not None:
            t = tu if keep is None else tu[keep]
            Eh = net(t).squeeze(-1)
            Th = library(t[:, 0], t[:, 1], Eh)
            r = (Th * ct).sum(-1)
            if keep is not None:                          # ライブラリ段の 税関 (E² 爆発 等)
                r = r[torch.isfinite(r)]
            l = l + lam * r.pow(2).mean()
        return l
    train(loss, net, steps=2000)
    with torch.no_grad():
        pred = net(torch.tensor(np.stack([mt, pt], 1), device=DEV)).squeeze(-1)
        pred = pred.cpu().numpy()
    if not np.isfinite(pred).all():
        return np.inf                                     # 死んだ 学習は 死んだと 報告
    return float(np.sqrt(((pred - Et) ** 2).mean()))


def phase_b():
    print("\nPhase B — 発見した 法則を 無ラベルの 先生に (E=√(m²+p²)・8 シード)")
    c8, gap8 = discover_c(*make_data(0, 8)[0])
    print(f"  N=8 の 零空間発見: c = {np.round(c8, 3)} (真値 [0,0,0,0,−1,−1,1]・ギャップ {gap8:.0f})")
    print(f"  {'N':>4} {'基準(中央値/最悪)':>22} {'発見則':>18} {'真値則(天井)':>18}")
    for n_lab in (8, 32, 128):
        res = {}
        for arm in ("baseline", "discovered", "oracle"):
            r = sorted(fit(s, n_lab, arm) for s in range(8))
            res[arm] = (r[4], r[-1])
        print(f"  {n_lab:>4} "
              f"{res['baseline'][0]:>11.4f}/{res['baseline'][1]:>8.4f} "
              f"{res['discovered'][0]:>9.4f}/{res['discovered'][1]:>8.4f} "
              f"{res['oracle'][0]:>9.4f}/{res['oracle'][1]:>8.4f}")


def phase_c():
    print("\nPhase C — 無ラベル集合を 汚染 (5% Inf/NaN + 2.5% 巨大値)・N=32・8 シード")
    for honesty, lab in (("ieee", "IEEE 素通し"), ("flags", "フラグ除外")):
        r = [fit(s, 32, "discovered", contaminate=True, honesty=honesty)
             for s in range(8)]
        dead = sum(1 for v in r if not np.isfinite(v))
        alive = sorted(v for v in r if np.isfinite(v))
        med = alive[len(alive) // 2] if alive else float("nan")
        print(f"  {lab:<10}: 死亡 {dead}/8"
              + (f"  生存の 中央値 RMSE {med:.4f}" if alive else "  (全滅)"))
    r0 = sorted(fit(s, 32, "discovered") for s in range(8))
    print(f"  (参考: 汚染なし 発見則 中央値 {r0[4]:.4f})")


if __name__ == "__main__":
    phase_a()
    phase_b()
    phase_c()
