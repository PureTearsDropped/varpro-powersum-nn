#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""implicit_transformer — 本物の Transformer の FNN ブロックに 発見装置を 向ける (言語編・その2)。

  モデル: 最小の GPT (2層・4ヘッド・d=64・文脈64・tied 埋め込み)。コーパスは CORPUS 環境変数。
  ① 学習して bpc を FNN-only 基準線 (implicit_charlm: 2.45) と 並べる。
  ② 各層の FNN ブロックの 入出力ジェット (x = ln2 後の 入力, y = FFN 出力・残差加算前) を
     集め、暗黙法則 Θ(x,y)·c = 0 を 問う:
       (a) 線形законы — ギャップ + σ_min (「どれだけ 線形から 遠いか」の 連続量)
       (b) 二次法則 — PCA 主部分空間 (x8+y8) の 2次ライブラリで
     対照 2 本 (どちらも 誠実さの 装置):
       ・未学習 (乱数初期化) モデルの 同じ ジェット — 「学習が 法則を 作ったか」
       ・シャッフル対照 — y を 行シャッフルして x↔y の 対応を 壊す。壊しても 生き残る
         「法則」は x か y の 周辺だけの 退化で、FFN の 入出力関係では ない。
  規律: 見つからなければ 拒否と 言う。閾値 (ギャップ>10) は 事前固定。
"""
import math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CTX, D, NH, NL, FW, STEPS, BS = 64, 64, 4, 2, 256, 6000, 128


def load_corpus():
    path = os.environ.get("CORPUS", "corpus.txt")
    if not os.path.exists(path):
        sys.exit("CORPUS 環境変数で コーパスを 指定してください")
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    idx = {c: i for i, c in enumerate(chars)}
    return torch.tensor([idx[c] for c in text], dtype=torch.long, device=DEV), chars


class BilinFFN(nn.Module):
    "⊙サンドイッチ: W₃(W₁x ⊙ W₂x)。掛け算は 回路で 焼き、前後の 写像だけ 学ぶ。"
    def __init__(self, w, gated=False):
        super().__init__()
        self.a = nn.Linear(D, w)
        self.b = nn.Linear(D, w)
        self.o = nn.Linear(w, D)
        self.gated = gated                         # gated=True → GLU (⊙ の 片腕に GELU)

    def forward(self, x):
        u = self.a(x)
        v = self.b(x)
        return self.o(u * (F.gelu(v) if self.gated else v))


class Block(nn.Module):
    def __init__(self, ffn="gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        w2 = 2 * FW // 3                           # パラメタ数を ほぼ 揃える (2·D·FW ≈ 3·D·w2)
        self.ff = (nn.Sequential(nn.Linear(D, FW), nn.GELU(), nn.Linear(FW, D))
                   if ffn == "gelu" else BilinFFN(w2, gated=(ffn == "glu")))

    def forward(self, x, jets=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, NH, D // NH).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        h_in = self.ln2(x)
        h_out = self.ff(h_in)                       # ← 「LLM の FNN」の 入出力ジェット
        if jets is not None:
            jets.append((h_in.detach(), h_out.detach()))
        return x + h_out


class TinyGPT(nn.Module):
    def __init__(self, V, ffn="gelu"):
        super().__init__()
        self.E = nn.Parameter(0.1 * torch.randn(V, D))
        self.pos = nn.Parameter(0.02 * torch.randn(CTX, D))
        self.blocks = nn.ModuleList(Block(ffn) for _ in range(NL))
        self.lnf = nn.LayerNorm(D)

    def forward(self, ix, jets=None):
        x = self.E[ix] + self.pos[: ix.shape[1]]
        for b in self.blocks:
            x = b(x, jets)
        return self.lnf(x) @ self.E.T               # tied 読み出し


def windows(data, n, rng):
    i = torch.randint(0, len(data) - CTX - 1, (n,), generator=rng, device=DEV)
    ix = torch.stack([data[j:j + CTX] for j in i])
    y = torch.stack([data[j + 1:j + CTX + 1] for j in i])
    return ix, y


def train_model(tr, V, seed=0, steps=STEPS, ffn="gelu"):
    torch.manual_seed(seed)
    net = TinyGPT(V, ffn).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(steps):
        ix, y = windows(tr, BS, rng)
        opt.zero_grad()
        F.cross_entropy(net(ix).reshape(-1, V), y.reshape(-1)).backward()
        opt.step(); sch.step()
    return net


def collect_jets(net, va, n_win=200):
    rng = torch.Generator(device=DEV).manual_seed(7)
    ix, _ = windows(va, n_win, rng)
    jets = []
    with torch.no_grad():
        net(ix, jets)
    return [(x.reshape(-1, D).cpu().numpy().astype(np.float64),
             y.reshape(-1, D).cpu().numpy().astype(np.float64)) for x, y in jets]


def nullspace(Th, want_c=False):
    s = np.sqrt((Th ** 2).mean(0)) + 1e-30
    _, sv, Vt = np.linalg.svd(Th / s, full_matrices=False)
    gap = sv[-2] / max(sv[-1], 1e-30)
    if not want_c:
        return gap, sv[-1]
    c = Vt[-1] / s
    return gap, sv[-1], c / np.linalg.norm(c)


def quad_library(X, Y, k=8):
    "x, y を 各 PCA k 次元に 落として 2 次ライブラリ [1, z, z⊗z(上三角)] を 作る。"
    def pca(A):
        A = A - A.mean(0)
        _, _, Vt = np.linalg.svd(A[:4000], full_matrices=False)
        Z = A @ Vt[:k].T
        return Z / Z.std(0)
    z = np.concatenate([pca(X), pca(Y)], 1)
    n, m = z.shape
    cols = [np.ones(n)] + [z[:, i] for i in range(m)] + \
           [z[:, i] * z[:, j] for i in range(m) for j in range(i, m)]
    return np.stack(cols, 1)


def probe(name, jets, lns):
    print(f"   {name}:")
    for L, ((X, Y), (gam, bet)) in enumerate(zip(jets, lns)):
        n = len(X)
        gap_l, sm_l, c = nullspace(
            np.concatenate([X, Y, np.ones((n, 1))], 1), want_c=True)
        # 見つかった 線形法則の 解剖: 質量は どこに 住むか + LayerNorm の 予言と 照合
        mx, my = np.linalg.norm(c[:D]), np.linalg.norm(c[D:2 * D])
        pred = np.concatenate([1.0 / gam, np.zeros(D), [-(bet / gam).sum()]])
        pred /= np.linalg.norm(pred)
        cosln = abs(float(c @ pred))
        Ys = Y[np.random.default_rng(0).permutation(n)]
        gap_ls, _ = nullspace(np.concatenate([X, Ys, np.ones((n, 1))], 1))
        lin = (f"厳密法則 (x側 {mx:.2f}/y側 {my:.2f}・シャッフル後も {gap_ls:.0f}"
               f"=xの構造・LayerNorm予言との cos {cosln:.4f})" if gap_l > 10 else "拒否")
        gap_q, _ = nullspace(quad_library(X, Y))
        gap_s, _ = nullspace(quad_library(X, Ys))
        quad = ("法則あり" if gap_q > 10 and gap_s < 10 else
                "周辺の退化" if gap_q > 10 else "拒否")
        print(f"     層{L}: 線形 ギャップ{gap_l:.0f} → {lin}")
        print(f"          二次 ギャップ{gap_q:5.1f} (シャッフル対照 {gap_s:5.1f}) → {quad}")


def main():
    data, chars = load_corpus()
    V = len(chars)
    n_tr = int(0.9 * len(data))
    tr, va = data[:n_tr], data[n_tr:]
    print(f"corpus: {len(data):,} 文字・語彙 {V}")

    print(f"① TinyGPT (層{NL}・頭{NH}・d{D}・文脈{CTX}) を 学習")
    net = train_model(tr, V)
    rng = torch.Generator(device=DEV).manual_seed(1)
    with torch.no_grad():
        ix, y = windows(va, 500, rng)
        bpc = float(F.cross_entropy(net(ix).reshape(-1, V), y.reshape(-1))) / math.log(2)
    print(f"   held-out {bpc:.2f} bpc  (FNN-only 基準線 2.45 / ユニグラム 4.77)")
    with torch.no_grad():                          # 生成見本
        s = list(windows(va, 1, rng)[0][0].cpu().numpy())
        for _ in range(300):
            lg = net(torch.tensor([s[-CTX:]], device=DEV))[0, -1] / 0.8
            s.append(int(torch.multinomial(F.softmax(lg, -1), 1, generator=rng)))
        print("   見本:", repr("".join(chars[i] for i in s[CTX:]))[:200])

    print("② FNN ブロックの ジェットに 暗黙法則を 問う (閾値: ギャップ>10・事前固定)")
    def ln_params(m):
        return [(b.ln2.weight.detach().cpu().numpy().astype(np.float64),
                 b.ln2.bias.detach().cpu().numpy().astype(np.float64))
                for b in m.blocks]
    probe("学習済み", collect_jets(net, va), ln_params(net))
    fresh = TinyGPT(V).to(DEV)                     # 対照: 未学習の 同型モデル
    probe("未学習 (乱数)", collect_jets(fresh, va), ln_params(fresh))


def ffn_shootout():
    "FFN の 焼き加減 対決: GELU MLP / ⊙サンドイッチ (活性化なし) / GLU。2 シード。"
    data, chars = load_corpus()
    V = len(chars)
    n_tr = int(0.9 * len(data))
    tr, va = data[:n_tr], data[n_tr:]
    print("FFN 対決 (パラメタ ほぼ 同数・2 シード・held-out bpc)")
    for ffn, lab in (("gelu", "GELU MLP (基準)"), ("bilinear", "⊙サンドイッチ (活性化なし)"),
                     ("glu", "GLU (⊙+ゲート)")):
        bpcs = []
        for seed in (0, 1):
            net = train_model(tr, V, seed=seed, ffn=ffn)
            rng = torch.Generator(device=DEV).manual_seed(99)
            with torch.no_grad():
                ix, y = windows(va, 500, rng)
                bpcs.append(float(F.cross_entropy(net(ix).reshape(-1, V),
                                                  y.reshape(-1))) / math.log(2))
        print(f"  {lab:<24} {bpcs[0]:.3f} / {bpcs[1]:.3f} bpc")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ffn":
        ffn_shootout()
    else:
        main()
