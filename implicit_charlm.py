#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""implicit_charlm — 本物の char-LM の FNN に 発見装置を 向ける (implicit_* 言語編・その1)。

  モデル = tied 埋め込み LM の 最小形 (GPT-2 の 結線の 玩具):
      前 n 文字の 埋め込みを 連結 → FNN (MLP) → 出力は Eᵀ 読み出し (入口と 同じ 表)
  コーパスは データ非依存 (環境変数 CORPUS のみ・本文の ハードコード なし)。

  ① 学習: held-out bits/char を 一様・ユニグラムの 基準と 並べて 報告 (+短い 生成見本)。
  ② 発見装置を 学習後の ジェットへ:
       (a) 埋め込み表 E の 線形零空間 — 「言語の 表は 低ランク」を ギャップ 証明書で
       (b) 合成器の 出力 h(ctx) は 埋め込み多様体に 住むか — cos(h, E[真の次]) の 分布
  規律: 見つからない ものは 見つからないと 言う (ギャップが 小さければ 拒否)。
"""
import math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
N_CTX, D, WIDTH, STEPS, BATCH = 8, 64, 512, 4000, 2048


def load_corpus():
    path = os.environ.get("CORPUS", "corpus.txt")
    if not os.path.exists(path):
        sys.exit("CORPUS 環境変数で コーパスを 指定してください (本文は ハードコードしない)")
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    idx = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([idx[c] for c in text], dtype=torch.long, device=DEV)
    return data, chars


class TiedFnnLM(nn.Module):
    "logits = FNN(concat(E[前n文字])) @ Eᵀ — tied 埋め込みの 最小 LM。"
    def __init__(self, V):
        super().__init__()
        self.E = nn.Parameter(0.1 * torch.randn(V, D))
        self.f = nn.Sequential(nn.Linear(N_CTX * D, WIDTH), nn.GELU(),
                               nn.Linear(WIDTH, WIDTH), nn.GELU(),
                               nn.Linear(WIDTH, D))

    def hidden(self, ctx):                        # ctx: (B, N_CTX) 整数
        return self.f(self.E[ctx].reshape(ctx.shape[0], -1))

    def forward(self, ctx):
        return self.hidden(ctx) @ self.E.T


def batches(data, n, rng):
    i = torch.randint(0, len(data) - N_CTX - 1, (n,), generator=rng, device=DEV)
    ctx = torch.stack([data[i + k] for k in range(N_CTX)], 1)
    return ctx, data[i + N_CTX]


def main():
    data, chars = load_corpus()
    V = len(chars)
    n_tr = int(0.9 * len(data))
    tr, va = data[:n_tr], data[n_tr:]
    print(f"corpus: {len(data):,} 文字・語彙 {V}  (train {n_tr:,} / val {len(va):,})")

    # ---- 基準線
    cnt = torch.bincount(tr, minlength=V).double()
    p = cnt / cnt.sum()
    uni_bpc = float(-(p[p > 0] * p[p > 0].log2()).sum())
    print(f"① 学習 — 基準線: 一様 {math.log2(V):.2f} bpc / ユニグラム {uni_bpc:.2f} bpc")

    torch.manual_seed(0)
    net = TiedFnnLM(V).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = torch.Generator(device=DEV).manual_seed(0)
    for step in range(STEPS):
        ctx, y = batches(tr, BATCH, rng)
        opt.zero_grad()
        F.cross_entropy(net(ctx), y).backward()
        opt.step(); sch.step()
    with torch.no_grad():
        ctx, y = batches(va, 20000, rng)
        bpc = float(F.cross_entropy(net(ctx), y)) / math.log(2)
    print(f"   tied-FNN (n={N_CTX}, d={D}): held-out {bpc:.2f} bpc")

    with torch.no_grad():                          # 生成見本 (貪欲でなく 温度 0.8)
        s = list(batches(va, 1, rng)[0][0].cpu().numpy())
        for _ in range(120):
            logits = net(torch.tensor([s[-N_CTX:]], device=DEV))[0] / 0.8
            s.append(int(torch.multinomial(F.softmax(logits, -1), 1)))
        print("   見本:", repr("".join(chars[i] for i in s[N_CTX:]))[:120])

    # ---- ② 発見装置を ジェットへ
    print("② 発見装置:")
    E = net.E.detach().cpu().numpy().astype(np.float64)
    s_ = np.sqrt((E ** 2).mean(0)) + 1e-30
    sv = np.linalg.svd(E / s_, compute_uv=False)
    gap = sv[-2] / max(sv[-1], 1e-30)
    eff = int((sv > 0.01 * sv[0]).sum())
    q = np.percentile(sv, [0, 25, 50, 75, 100])
    print(f"   (a) 埋め込み表 E ({V}×{D}) の 線形零空間: σ_min ギャップ {gap:.2f}"
          f" → {'法則あり' if gap > 10 else '拒否 (厳密な 線形法則は ない)'}"
          f" / 実効ランク {eff}/{min(V, D)} (σ > 1% σ_max)"
          f" / σ 分位 [{q[0]:.2f} {q[1]:.2f} {q[2]:.2f} {q[3]:.2f} {q[4]:.2f}]")

    with torch.no_grad():
        ctx, y = batches(va, 8000, rng)
        h = net.hidden(ctx)
        En = F.normalize(net.E, dim=1)
        hn = F.normalize(h, dim=1)
        cos_true = (hn * En[y]).sum(1)
        j = torch.randint(0, V, y.shape, generator=rng, device=DEV)
        cos_rand = (hn * En[j]).sum(1)
    print(f"   (b) 合成器の 出力は 埋め込み多様体に 住むか:"
          f" cos(h, E[真の次]) 中央値 {cos_true.median():.3f}"
          f" vs cos(h, E[乱択]) 中央値 {cos_rand.median():.3f}")
    print("done — 次: この ジェットに 二次以上の 暗黙法則を 問う / 見つかった 法則を 正則化に")


if __name__ == "__main__":
    main()
