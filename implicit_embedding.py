#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""implicit_embedding — 埋め込み空間の 暗黙法則: 公理を 先生に・法則を 証明書つきで 測る。

  背景: 構造を **アーキテクチャに** 課すのは 中立か 損 (bitter lesson・実測済)。
  本実験の 問い: 同じ 構造を **法則 (左辺=0 の 損失) として** 課すと どうなるか。

  課題: ℤ/16 加法の 次トークン予測 (a, b → a+b)。tied 埋め込み E (16×16) +
  合成器 h = MLP。ラベルは 全 256 ペアの 一部 (frac) だけ。

  ① 公理を 無ラベルの 先生に (8 シード × frac 3 水準):
       可換律   h(x,y) − h(y,x) = 0            (無ラベルペアで)
       結合律   h(h(x,y),z) − h(x,h(y,z)) = 0  (ランダム三つ組で)
       単位元   h(E[0],x) − x = 0
     公理は 課題と 一致する 構造 (アーベル群) — meta 法則の 予言は「一致×希少なら 勝つ」。
  ② 発見装置を 埋め込みへ: 全データ学習後の E に 加法法則
       E[a] + E[b] − E[a+b] = 0 ?
     を 零空間で 問う。ギャップが 小さければ 装置は 正しく 拒否する
     (フーリエ型の 表は 線形加法では ない — 拒否も 結果)。

  規律: 8 シード + 最悪ケース・負けも そのまま 印字。λ は 未調整 (1.0/0.3/0.3 固定)。
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
M = 16


def fourier_table():
    "ℤ/16 の 指標表 (実フーリエ): 課題と 一致する 固定 基底 — m 値の 法則の 巡回版。"
    a = torch.arange(M, dtype=torch.float64)
    cols = []
    for k in range(M // 2):
        cols.append(torch.cos(2 * math.pi * k * a / M))
        cols.append(torch.sin(2 * math.pi * k * a / M))
    return torch.stack(cols, 1)


class Model(nn.Module):
    def __init__(self, d=16, width=64, basis="free"):
        super().__init__()
        if basis == "fourier":
            self.register_buffer("E", fourier_table().to(DEV))
        else:
            self.E = nn.Parameter(0.3 * torch.randn(M, d))
        self.h = nn.Sequential(nn.Linear(2 * d, width), nn.Tanh(),
                               nn.Linear(width, width), nn.Tanh(),
                               nn.Linear(width, d))

    def compose(self, x, y):
        return self.h(torch.cat([x, y], -1))

    def logits(self, a, b):
        return self.compose(self.E[a], self.E[b]) @ self.E.T


def run_one(seed, frac, axioms, basis="free"):
    rs = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(M) for b in range(M)])
    perm = rs.permutation(len(pairs))
    n_tr = int(frac * len(pairs))
    tr, te = pairs[perm[:n_tr]], pairs[perm[n_tr:]]
    ta = torch.tensor(tr[:, 0], device=DEV); tb = torch.tensor(tr[:, 1], device=DEV)
    tc = (ta + tb) % M
    ea = torch.tensor(te[:, 0], device=DEV); eb = torch.tensor(te[:, 1], device=DEV)
    ec = (ea + eb) % M

    torch.manual_seed(seed)
    net = Model(basis=basis).to(DEV)
    # AdamW + weight decay + 長め: 群課題の 汎化は grokking 域 (wd なしだと 基準腕が
    # 記憶のまま 終わり 不当に 弱く 見える) — 両腕 同条件
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(8000):
        opt.zero_grad()
        loss = F.cross_entropy(net.logits(ta, tb), tc)
        if axioms:
            i = torch.randint(0, M, (256,), generator=g, device=DEV)
            j = torch.randint(0, M, (256,), generator=g, device=DEV)
            k = torch.randint(0, M, (256,), generator=g, device=DEV)
            x, y, z = net.E[i], net.E[j], net.E[k]
            comm = (net.compose(x, y) - net.compose(y, x)).pow(2).mean()
            asc = (net.compose(net.compose(x, y), z)
                   - net.compose(x, net.compose(y, z))).pow(2).mean()
            unit = (net.compose(net.E[0].expand_as(x), x) - x).pow(2).mean()
            loss = loss + 1.0 * comm + 0.3 * asc + 0.3 * unit
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((net.logits(ea, eb).argmax(-1) == ec).float().mean())
    return acc, net


ARMS = [("free", False, "基準"), ("free", True, "公理"),
        ("fourier", False, "フーリエ基底"), ("fourier", True, "基底+公理")]

def part1():
    print("① 課し場所の 対決 (ℤ/16 加法・テスト正解率 中央値/最悪・8 シード)")
    print("   基準=自由埋め込み / 公理=損失に法則 / フーリエ基底=一致する入力基底(指標表)")
    hdr = " ".join(f"{lab:>16}" for _, _, lab in ARMS)
    print(f"   {'frac':>5} {hdr}")
    for frac in (0.15, 0.3, 0.5):
        cells = []
        for basis, ax, _ in ARMS:
            accs = sorted(run_one(s, frac, ax, basis)[0] for s in range(8))
            cells.append(f"{accs[4]:>8.3f}/{accs[0]:>5.3f}")
        print(f"   {frac:>5} " + "  ".join(cells))


def part2():
    print("\n② 発見装置を 埋め込みへ: E[a]+E[b]−E[a+b] = 0 は 成り立つか (全データ学習後)")
    _, net = run_one(0, 1.0, False)
    E = net.E.detach().cpu().numpy()
    rows = []
    for a in range(M):
        for b in range(M):
            for k in range(E.shape[1]):
                rows.append([E[a, k], E[b, k], E[(a + b) % M, k], 1.0])
    Th = np.array(rows)
    s = np.sqrt((Th ** 2).mean(0)) + 1e-30
    _, sv, Vt = np.linalg.svd(Th / s, full_matrices=False)
    gap = sv[-2] / max(sv[-1], 1e-30)
    c = Vt[-1] / s                                # 列正規化を 戻す (元の 列の 係数で 表示)
    c = c / np.abs(c).max()
    verdict = "法則あり" if gap > 10 else "拒否 (この 表は 線形加法では ない)"
    print(f"   零空間: c = {np.round(c, 3)}  ギャップ {gap:.1f} → {verdict}")
    # 陽性対照: 本当に 加法的な 表 E[a] = a·v (w≠0 だと 次元ごとに 切片が 違い
    # 共有 c では 法則に ならない — それも 装置は 正しく 弱い ギャップで 言う) なら 拾うか
    rs = np.random.default_rng(0)
    v = rs.standard_normal(4)
    rows = []
    for a in range(M):
        for b in range(M):
            for k in range(4):
                rows.append([a * v[k], b * v[k], (a + b) * v[k], 1.0])
    Th = np.array(rows)
    s = np.sqrt((Th ** 2).mean(0)) + 1e-30
    _, sv, Vt = np.linalg.svd(Th / s, full_matrices=False)
    gap2 = sv[-2] / max(sv[-1], 1e-30)
    c2 = Vt[-1] / s
    c2 = c2 / np.abs(c2).max()
    print(f"   陽性対照 (真に 加法的な 表 E[a]=a·v): c = {np.round(c2, 2)}  ギャップ {gap2:.0e}"
          f" → c=(1,1,−1,0) を 拾う ⟹ 装置は 見える ものは 見る")


if __name__ == "__main__":
    part1()
    part2()
