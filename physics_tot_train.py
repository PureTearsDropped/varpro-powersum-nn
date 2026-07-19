#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""物理方程式学習 × CUDA 全域算術 — 「NaN 毒で 学習が 全滅しない」を 実ワークロードで。

  課題: 重力の 逆二乗則 F = G·m₁·m₂ / r²（unified_physics_learning.py の 実験1）。
  構造化モデル: F̂ = exp(g) · m₁^a · m₂^b / r^c（指数 a,b,c と logG=g を 学習。真値 a=b=1, c=2）。

  毒: 実データに r≈1e-20 の 汚染行を 混ぜる（センサ異常の 模型）。
    r^c → underflow → 0 → 除算 → **IEEE: inf → 勾配 NaN → 重み恒久汚染 = 学習全体が 死ぬ**
    全域算術(TOT): r^c → ε(=±MIN·LE) → 除算 → ±MAX·GE。**NaN 出さず・フラグが 汚染行を 名指し**。

  3 腕 × 8 シード（handoff 規律: 8 seeds + worst case）:
    IEEE      : 素の float32
    TOT       : 全域算術（飽和±MAX/ε・a/0=0・フラグ）
    TOT+mask  : 同上 ＋ フラグ行を 損失から 除外（フラグは 実行時に 使える 情報）
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F32 = torch.float32
MAX = torch.finfo(F32).max
MIN = torch.finfo(F32).tiny
GE, LE = 1, 2


def sat(raw, flag_acc):
    """全域化: 溢れ→±MAX(GE)・潰れ→±MIN=ε(LE)。フラグを 蓄積。勾配は 非飽和枝を 流れる。"""
    sign = torch.sign(raw.detach())
    a = raw.abs()
    over = (a > MAX) | torch.isinf(raw)
    under = (a > 0) & (a < MIN)
    out = torch.where(over, sign * MAX, raw)
    out = torch.where(under, sign * MIN, out)
    flag_acc |= over.to(torch.uint8) * GE
    flag_acc |= under.to(torch.uint8) * LE
    return out, flag_acc

def _safe_div_val(num, den):
    """値の 全域除算（勾配なし の 素関数）: b=0→0・飽和 ±MAX・潰れ ±MIN。"""
    bz = (den == 0)
    raw = num / torch.where(bz, torch.ones_like(den), den)
    raw = torch.where(bz, torch.zeros_like(raw), raw)
    sign = torch.sign(raw); a = raw.abs()
    raw = torch.where((a > MAX) | torch.isinf(raw), sign * MAX, raw)
    raw = torch.where((a > 0) & (a < MIN), sign * MIN, raw)
    return raw

GRAD_EVENT = {}                                       # 直近 backward の 勾配飽和 行（勾配フラグ）

class SatDiv(torch.autograd.Function):
    """全域除算 — **逆伝播も 全域算術**。
       診断で 判明: 順伝播だけ 全域化しても、backward の −num/den² が den²=underflow→0→inf で
       勾配が 毒される（逆伝播も 算術・そこにも a/0 と 溢れが 住む）。
       ⟹ d/dnum = 1/den, d/dden = −(num/den)/den を **各段 全域除算**で 計算（inf 不生成）。"""
    @staticmethod
    def forward(ctx, num, den):
        ctx.save_for_backward(num, den)
        return _safe_div_val(num, den)
    @staticmethod
    def backward(ctx, g):
        num, den = ctx.saved_tensors
        inv = _safe_div_val(torch.ones_like(den), den)     # 1/den（全域）
        q = _safe_div_val(num, den)                        # num/den（全域）
        qd = _safe_div_val(q, den)                         # q/den（全域 ⟹ ±MAX 止まり）
        GRAD_EVENT['rows'] = (qd.abs() >= MAX * 0.99) | (inv.abs() >= MAX * 0.99)
        dden = -g * qd
        return g * inv, dden

def tot_div(num, den, flag_acc):
    """全域除算（フラグ付き・勾配も 全域）。"""
    bz = (den == 0)
    raw_probe = num.detach() / torch.where(bz, torch.ones_like(den), den.detach())
    over = torch.isinf(raw_probe) | (raw_probe.abs() > MAX)
    under = (raw_probe.abs() > 0) & (raw_probe.abs() < MIN) & ~bz
    flag_acc |= over.to(torch.uint8) * GE
    flag_acc |= under.to(torch.uint8) * LE
    return SatDiv.apply(num, den), flag_acc


def forward(params, X, mode):
    """F̂ = exp(g)·m₁^a·m₂^b / r^c。mode='ieee' は 素・'tot' は 全域算術。"""
    g, a, b, c = params
    m1, m2, r = X[:, 0], X[:, 1], X[:, 2]
    flag = torch.zeros(X.shape[0], dtype=torch.uint8, device=X.device)
    if mode == "ieee":
        num = torch.exp(g) * m1.pow(a) * m2.pow(b)
        return num / r.pow(c), flag
    num, flag = sat(torch.exp(g) * m1.pow(a), flag)
    num, flag = sat(num * m2.pow(b), flag)
    den, flag = sat(r.pow(c), flag)                       # r≈0 ⟹ ここで ε(LE)
    out, flag = tot_div(num, den, flag)                   # ε で 割る ⟹ ±MAX(GE)
    return out, flag


def run_one(seed, mode, n_clean=256, n_poison=4, iters=2000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    G = 6.674e-2                                          # スケーリング済み G（学習しやすい 大きさ）
    m1 = rng.uniform(0.5, 2.0, n_clean); m2 = rng.uniform(0.5, 2.0, n_clean)
    r = rng.uniform(0.5, 2.0, n_clean)
    Xc = np.stack([m1, m2, r], 1)
    yc = G * m1 * m2 / r**2
    # 汚染行: r ≈ 1e-20（センサ異常）。目標値は 記録系の 飽和値（±MAX 相当）
    Xp = np.stack([rng.uniform(0.5, 2, n_poison), rng.uniform(0.5, 2, n_poison),
                   np.full(n_poison, 1e-20)], 1)
    yp = np.full(n_poison, MAX)
    X = torch.tensor(np.vstack([Xc, Xp]), dtype=F32, device=DEV)
    y = torch.tensor(np.concatenate([yc, yp]), dtype=F32, device=DEV)
    poison_idx = torch.arange(n_clean, n_clean + n_poison, device=DEV)

    params = [torch.tensor(v, dtype=F32, device=DEV, requires_grad=True)
              for v in (0.0, 0.5, 0.5, 1.0)]              # g,a,b,c 初期値
    opt = torch.optim.Adam(params, lr=0.03)
    huber = torch.nn.HuberLoss(delta=1.0)
    nan_events = 0
    tainted = torch.zeros(X.shape[0], dtype=torch.bool, device=DEV)   # 蓄積 汚染フラグ(順+勾配)
    for _ in range(iters):
        opt.zero_grad()
        pred, flag = forward(params, X, "ieee" if mode == "ieee" else "tot")
        tainted |= (flag > 0)
        if mode == "tot_mask":
            keep = ~tainted                               # これまでに フラグが 立った行を 除外
            loss = huber(torch.log1p(pred[keep].abs()) * torch.sign(pred[keep]),
                         torch.log1p(y[keep].abs()) * torch.sign(y[keep]))
        else:
            loss = huber(torch.log1p(pred.abs()) * torch.sign(pred),
                         torch.log1p(y.abs()) * torch.sign(y))
        loss.backward()
        if mode != "ieee" and 'rows' in GRAD_EVENT:
            tainted |= GRAD_EVENT['rows']                 # **勾配フラグ**（逆伝播の 飽和イベント）
        if any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() for p in params if p.grad is not None):
            nan_events += 1                                # IEEE は ここで 毒が 入る
        opt.step()
    with torch.no_grad():
        vals = [float(p) for p in params]
        dead = any(np.isnan(v) or np.isinf(v) for v in vals)
        # 指数の 回復誤差（真値 a=1,b=1,c=2）
        err = np.nan if dead else float(abs(vals[1]-1) + abs(vals[2]-1) + abs(vals[3]-2))
        # フラグ（順+勾配・蓄積）は 汚染行を 名指ししたか
        hit = int(tainted[poison_idx].sum()) if mode != "ieee" else -1
        false_pos = int(tainted[:n_clean].sum()) if mode != "ieee" else -1
    return dict(dead=dead, err=err, nan_events=nan_events, hit=hit, fp=false_pos)


def main():
    print(f"device: {DEV} ({torch.cuda.get_device_name(0) if DEV.type=='cuda' else 'CPU'})")
    print("=" * 78)
    print("重力則 F=G·m₁m₂/r² の 学習 — 汚染行(r≈1e-20) 4/260 混入・8 シード")
    print("=" * 78)
    arms = [("tot_clean", "TOT 汚染なし(基準)"), ("ieee", "IEEE float32"),
            ("tot", "全域算術(TOT)"), ("tot_mask", "TOT+フラグ行除外")]
    for mode, name in arms:
        if mode == "tot_clean":
            results = [run_one(s, "tot", n_poison=0) for s in range(8)]
        else:
            results = [run_one(s, mode) for s in range(8)]
        deads = sum(r['dead'] for r in results)
        errs = [r['err'] for r in results if not r['dead']]
        nanev = sum(r['nan_events'] for r in results)
        line = f"  {name:<18} 死亡 {deads}/8"
        if errs:
            line += f"  指数誤差 中央値 {np.median(errs):.3f} 最悪 {max(errs):.3f}"
        else:
            line += "  指数誤差 —（全滅）"
        line += f"  NaN勾配 {nanev} 回"
        if mode != "ieee":
            hits = [r['hit'] for r in results]; fps = [r['fp'] for r in results]
            line += f"  フラグ命中 {min(hits)}〜{max(hits)}/4 誤検 {max(fps)}"
        print(line)
    print()
    print("  ⟹ 期待: IEEE は 汚染行の inf→NaN勾配で 死ぬ／TOT は 生存し フラグが 汚染行を 名指し。")
    print("     （結果が 期待と 違えば それも そのまま 報告する）")


if __name__ == "__main__":
    main()
