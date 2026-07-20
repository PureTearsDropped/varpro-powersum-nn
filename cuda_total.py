#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証
"""CUDA(torch) 全域算術 + 配線表ライブラリ — GPU 版の「配線＝計算」。

  ・数 = (val: float32, flag: uint8)。flag ビット: GE=1(≥) LE=2(≤) SUNK=4(符号不明)。
  ・全域化: overflow→±MAX+GE / underflow→±MIN(=ε・向き保持)+LE / a/0=0 / **NaN·Inf は 決して 出さない**。
  ・配線表 = 構造テンソル T[k,i,j]（σ(i,j)·δ_{k,i∘j}）。**T を 差し替えると 同じカーネルが
    複素/四元/セデニオン/行列積/畳み込みに 変わる**（wiring_registry の GPU 版）。
  ・群積/MAC は float64 で 貯めて **飽和（丸め）は 最後に 1 回**（multi_add/mul_fused の 哲学）。

  **ゲート版との 意味論の違い（正直な 但し書き）**: フラグは 全域化イベント（飽和±MAX/ε=±MIN/
  0除算/相殺）**のみ**。float32 の 通常丸め（最近接）は フラグしない — ゲート版は 切り捨てごとに
  ge を 立てるが、float の 最近接丸めは 方向を 持たないため 片側境界に できない。
  実測: 「フラグなしで float64 真値と 違う」320,019 件は 全件 float32 丸めで 説明・飽和フラグの嘘 0。
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

GE, LE, SUNK = 1, 2, 4
F32 = torch.float32
MAX = torch.finfo(F32).max            # 飽和天井
MIN = torch.finfo(F32).tiny           # ε = 最小正規数（向き付き 無限小）


def _sat(raw64, dev):
    """float64 の 生値 → (float32 値, フラグ)。溢れ→±MAX+GE / 潰れ→±MIN+LE / NaN 出さない。
       NaN 入力は (0, 境界なし+SUNK) に 全域化（外部AI監査 2026-07-19 の 指摘で 追加）。"""
    nan = torch.isnan(raw64)
    raw64 = torch.where(nan, torch.zeros_like(raw64), raw64)
    sign = torch.sign(raw64)
    a = raw64.abs()
    over = a > MAX                                   # inf も ここで 捕まる
    under = (a > 0) & (a < MIN)
    val = raw64.clone()
    val = torch.where(over, sign * MAX, val)
    val = torch.where(under, sign * MIN, val)
    flag = torch.zeros(raw64.shape, dtype=torch.uint8, device=dev)
    flag |= over.to(torch.uint8) * GE
    flag |= under.to(torch.uint8) * LE
    flag |= nan.to(torch.uint8) * (GE | LE | SUNK)
    return val.to(F32), flag


class Tot:
    """全域数のテンソル: val float32・flag uint8（同形）。

       flag=None（利用者の 入口）では 入力を **全域化してから** 受け入れる:
       NaN→(0, 境界なし+SUNK) / ±Inf・float32範囲外→±MAX+GE / 非正規化数→±MIN+LE。
       「NaN/Inf を 決して 作らない」は この入口で 初めて 不変条件になる
       （外部AI監査 2026-07-19: 旧版は 入口が 素通しで 0×Inf=NaN の 経路が あった）。

       **flag を 明示的に 渡す 経路は 内部用（unsafe）**: val が 全域化済みであることは
       呼び出し側の 責任で、検査しない（演算関数が 生成する 値は 常に 全域化済み）。
       外部からは Tot(x) の 1 引数形を 使うこと。"""
    __slots__ = ('val', 'flag')
    def __init__(self, val, flag=None):
        if flag is None:
            self.val, self.flag = _sat(val.double(), val.device)
        else:
            self.val = val.to(F32)
            self.flag = flag
    @property
    def device(self): return self.val.device


def _mul_flags(fa, fb):
    """積の フラグ合成（E1 保守則）: ≥·≥=≥, ≤·≤=≤, =·x=x, ≥·≤=境界なし。SUNK は 伝播。"""
    ga, la = fa & GE, (fa >> 1) & 1
    gb, lb = fb & GE, (fb >> 1) & 1
    ge_o = ((ga | gb) & ~(la | lb)) & 1            # どちらかが ≥・どちらも ≤ でない
    le_o = ((la | lb) & ~(ga | gb)) & 1
    nb = ((ga | gb) & (la | lb)) & 1               # 混在 → 境界なし
    out = (ge_o * GE) | (le_o * LE) | (nb * (GE | LE))
    out = out.to(torch.uint8) | ((fa | fb) & SUNK)
    return out

def _danger_zero(v, f):
    """危険な0 = 表示0 かつ GEビット（真値の 大きさ・符号が 自由）。_sat(NaN) が この形。"""
    return (v == 0) & ((f & GE) > 0)

def _true_zero(v, f):
    """本当の0 = 表示0 かつ GEビットなし（|真|≤0 ⟹ 真=0）。**符号なし** — 向きは ±MIN が 運ぶ。"""
    return (v == 0) & ((f & GE) == 0)

def tot_mul(a, b):
    raw = a.val.double() * b.val.double()
    # x×0=0 厳密（IEEE で 0×inf は 出ない: 入力に inf が 無い 不変条件）
    val, sflag = _sat(raw, a.device)
    f = sflag | _mul_flags(a.flag, b.flag)
    # 危険な0 は 符号も 不明 — group_mul と 同一意味論を スカラーにも
    # （第4ラウンド監査 2026-07-19: (0,GE)×(3,=) の 真の積は ±6 なのに SUNK が 欠けていた）。
    danger = _danger_zero(a.val, a.flag) | _danger_zero(b.val, b.flag)
    f = f | danger.to(torch.uint8) * SUNK
    # 本当の0（符号なし）は 万物を 吸収: (0,厳密)×(x,任意フラグ) の 真の積は 厳密に 0
    # ⟹ 出力は (0, フラグなし)。相手の 不確かさは 消える（利用者の 教義: x×0=0 厳密）。
    tz = _true_zero(a.val, a.flag) | _true_zero(b.val, b.flag)
    return Tot(val, torch.where(tz, torch.zeros_like(f), f))

def tot_add(a, b):
    """加算の フラグ則（2026-07-19 改訂・外部AI監査の 反例 (+MIN,LE)+(−MIN,=)→(0,LE) を 受けて）:
       同符号（符号既知）なら 単純和が 健全（|和|=|a|+|b| は 単調: ≥+≥=≥, ≤+≤=≤）。
       **相殺が 起こりうる**（異符号・どちらか0・符号不明）とき、入力に 境界が 一つでも あれば
       片側境界は 維持できない → 境界なし+符号不明。旧 clash 則（両GE異符号のみ）は これに 包含される。"""
    raw = a.val.double() + b.val.double()
    val, sflag = _sat(raw, a.device)
    fin = a.flag | b.flag
    sign_known = (fin & SUNK) == 0
    same_sign = (torch.sign(a.val) * torch.sign(b.val)) > 0        # 厳密（0 は 同符号に 含めない）
    cancel = ~(sign_known & same_sign)
    # 相殺可能 × フラグあり → 境界なし+SUNK。境界(GE/LE)だけでなく **SUNK 単独でも 落とす**:
    # (2,SUNK)+(3,SUNK) の 真値は ±2±3 ⟹ |真| ∈ {1,5} で 大きさ厳密が 壊れる
    # （第3ラウンド監査の 指摘を 受けた 自前オラクル強化で 発見・2026-07-19）。
    f = torch.where((fin > 0) & cancel, torch.full_like(fin, GE | LE | SUNK), fin)
    return Tot(val, sflag | f)

def tot_div(a, b):
    bz = (b.val == 0)
    raw = a.val.double() / torch.where(bz, torch.ones_like(b.val), b.val).double()
    raw = torch.where(bz, torch.zeros_like(raw), raw)          # a/0 = 0（Moore–Penrose）
    val, sflag = _sat(raw, a.device)
    fin = a.flag | b.flag
    nb = (fin & (GE | LE)) > 0                                  # 入力に 境界 → 商は 保守的に 境界なし
    f = sflag | torch.where(nb, torch.full_like(fin, GE | LE), torch.zeros_like(fin)) \
        | (fin & SUNK)
    # 危険な0（分母: 表示0でも 真の分母≠0 なら 商は ±・分子: 真の商の 符号不明）→ SUNK
    # （第4ラウンド監査: (1,=)/(0,GE) の 真の商は ±0.5 なのに SUNK が 欠けていた）。
    danger = _danger_zero(a.val, a.flag) | _danger_zero(b.val, b.flag)
    return Tot(val, f | danger.to(torch.uint8) * SUNK)


# ---------------------------------------------------------------- 配線表（構造テンソル）
# Cayley–Dickson 構成（自己完結・nd_algebra と 同一規約: (a,b)(c,d) = (ac − d̄b, da + bc̄)）
def _cd_conj(x):
    n = len(x)
    if n == 1: return x.copy()
    h = n // 2
    return np.concatenate([_cd_conj(x[:h]), -x[h:]])

def _cd_prod(x, y):
    n = len(x)
    if n == 1: return x * y
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    return np.concatenate([_cd_prod(a, c) - _cd_prod(_cd_conj(d), b),
                           _cd_prod(d, a) + _cd_prod(b, _cd_conj(c))])

def cd_omega(M):
    """符号表 OMEGA[i,j] ∈ {−1,+1}, 経路 = i⊕j（XOR routing）。"""
    E = np.eye(M)
    OM = np.zeros((M, M), dtype=int)
    for i in range(M):
        for j in range(M):
            v = _cd_prod(E[i], E[j])
            k = int(np.argmax(np.abs(v)))
            assert k == (i ^ j), f"XOR routing 破れ M={M} ({i},{j})"
            OM[i, j] = int(np.sign(v[k]))
    return OM

def wiring_tensor(kind, M, device):
    """配線表 T[k,i,j]。kind: 'cd'（Cayley–Dickson XOR経路）/ 'cyclic'（巡回畳み込み）。"""
    T = torch.zeros(M, M, M, dtype=torch.float32, device=device)
    if kind == "cd":
        OM = cd_omega(M)
        for i in range(M):
            for j in range(M):
                T[i ^ j, i, j] = float(OM[i, j])
    elif kind == "cyclic":
        for i in range(M):
            for j in range(M):
                T[(i + j) % M, i, j] = 1.0
    else:
        raise ValueError(kind)
    return T

def group_mul(T, a, b):
    """配線積（バッチ）: c[...,k] = Σ_ij T[k,i,j]·a[...,i]·b[...,j]。
       float64 で 貯めて 飽和は 最後に 1 回（融合 MAC の 哲学）。"""
    raw = torch.einsum('kij,...i,...j->...k', T.double(), a.val.double(), b.val.double())
    val, sflag = _sat(raw, a.val.device)
    fin = (a.flag | b.flag)
    if int(fin.max()) == 0:                                     # 速い道: フラグなし
        return Tot(val, sflag)
    # パターン則（2026-07-19 監査後の 設計: 嘘ゼロは 絶対・その範囲で 最大限 残す）。
    # 出力成分 = 積の和。危険は「相殺」だけなので、成分ごとに 判定する:
    #   P0 関与項が 全て 厳密         → 主張そのまま
    #   P1 生きた項が 1 個            → 相殺不能 ⟹ スカラー積の E1 が そのまま生きる
    #                                    （SUNK でも 大きさ主張は 保持・符号のみ 不明）
    #   P2 生きた項が 全て 同符号(既知) → 和は 単調 ⟹ 全GE→GE / 全LE→LE・符号=共通
    #   P3/4 符号混在 or SUNK 多項     → 境界なし+SUNK（6−10 vs 6−2: 符号も 大きさも 落ちる）
    # 実測(パターン地図): 密±乱数は ほぼ P3/4（保守則が 最適）・疎積は 99% P1/P0・
    # 全正の畳み込みは 99% P2 — 疎・正値で 情報が 生き返る。
    Mk = T.shape[0]
    outf = torch.zeros_like(sflag)
    outsunk = torch.zeros(sflag.shape, dtype=torch.bool, device=val.device)
    NBv = torch.full_like(sflag[..., 0], GE | LE)
    for k in range(Mk):
        nz = (T[k] != 0).nonzero()
        if nz.numel() == 0:
            continue                                            # 空の出力行 = 常に 厳密な 0
        ii, jj = nz[:, 0], nz[:, 1]
        ss = torch.sign(T[k, ii, jj])
        ai, bj = a.val[..., ii], b.val[..., jj]
        fa, fb = a.flag[..., ii], b.flag[..., jj]
        # 「表示が0」≠「本当に0」（第3ラウンド監査 2026-07-19 の 指摘）。
        #   確実に0 = 表示0 かつ GEビットなし（|真|≤0 ⟹ 真=0）→ 死項として 除外してよい。
        #   危険な0 = 表示0 だが GEビットあり（真値は 任意/符号不明）→ 消してはならない:
        #             その項が 触る 成分は 境界なし+SUNK に 落とす。_sat(NaN)=(0,7) が この形。
        defz_a = (ai == 0) & ((fa & GE) == 0)
        defz_b = (bj == 0) & ((fb & GE) == 0)
        dead = defz_a | defz_b
        danger_t = ~dead & (((ai == 0) & ((fa & GE) > 0)) | ((bj == 0) & ((fb & GE) > 0)))
        danger = danger_t.any(dim=-1)
        live = ~dead & ~danger_t
        tf = torch.where(live, _mul_flags(fa, fb), torch.zeros_like(fa))   # 項ごとの E1
        touched = ((tf | (torch.where(live, fa | fb, torch.zeros_like(fa)))) > 0).any(dim=-1)
        sunk_any = ((((fa | fb) & SUNK) > 0) & live).any(dim=-1)
        n_live = live.sum(dim=-1)
        tsgn = ss * torch.sign(ai) * torch.sign(bj)
        smax = torch.where(live, tsgn, torch.full_like(tsgn, -2.0)).max(dim=-1).values
        smin = torch.where(live, tsgn, torch.full_like(tsgn, 2.0)).min(dim=-1).values
        same_sign = (smax == smin)
        ge_ok = ((tf & LE) == 0).all(dim=-1)                    # LE ビットの live 項が 無い
        le_ok = ((tf & GE) == 0).all(dim=-1)
        any_ge = ((tf & GE) > 0).any(dim=-1)
        any_le = ((tf & LE) > 0).any(dim=-1)
        f2 = (ge_ok & any_ge).to(torch.uint8) * GE | (le_ok & any_le).to(torch.uint8) * LE
        f2 = f2 | (~ge_ok & ~le_ok).to(torch.uint8) * (GE | LE)
        keep = (~sunk_any & (same_sign | (n_live <= 1))) | (sunk_any & (n_live == 1))
        p0 = ~touched
        fk = torch.where(p0, torch.zeros_like(f2), torch.where(keep, f2, NBv))
        sk = ~p0 & (~keep | (sunk_any & (n_live == 1)))
        fk = torch.where(danger, NBv, fk)                       # 危険な0 が 触った 成分
        sk = sk | danger
        outf[..., k] = fk
        outsunk[..., k] = sk
    f = sflag | outf | outsunk.to(torch.uint8) * SUNK
    return Tot(val, f)


# ---------------------------------------------------------------- 自己テスト
def self_test():
    import numpy as np
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'})")
    rng = np.random.default_rng(20260810)

    print("=" * 76)
    print("① 全域化: NaN/Inf を 決して 出さない・フラグは 嘘をつかない（敵対的）")
    print("=" * 76)
    N = 1_000_000
    # 敵対的入力: 極大・極小・ゼロ・普通 を 混ぜる
    pool = np.concatenate([
        rng.uniform(-1e38, 1e38, N // 4), rng.uniform(-1e-38, 1e-38, N // 4),
        np.zeros(N // 4), rng.standard_normal(N - 3 * (N // 4)) ])
    rng.shuffle(pool)
    av = torch.tensor(pool[:N], dtype=F32, device=dev)
    bv = torch.tensor(np.roll(pool[:N], 7), dtype=F32, device=dev)
    A, B = Tot(av), Tot(bv)
    bad_naninf = 0; lies = 0
    for name, op in [("mul", tot_mul), ("add", tot_add), ("div", tot_div)]:
        r = op(A, B)
        bad_naninf += int(torch.isnan(r.val).sum() + torch.isinf(r.val).sum())
        # 真値（float64・inf 可）と 照合: GE⟹|真|≥|表示|, LE⟹|真|≤|表示|, 無フラグ⟹一致
        a64, b64 = av.double(), bv.double()
        t = {"mul": a64 * b64, "add": a64 + b64,
             "div": torch.where(bv == 0, torch.zeros_like(a64), a64 / b64.where(b64 != 0, torch.ones_like(b64)))}[name]
        ge = (r.flag & GE) > 0; le = (r.flag & LE) > 0; ex = r.flag == 0
        lies += int((ge & ~le & (t.abs() < r.val.double().abs())).sum())        # GE: |真|≥|表示| か
        lies += int((le & ~ge & (t.abs() > r.val.double().abs())).sum())        # LE: |真|≤|表示| か
        # 無フラグ = float32 に 丸めた 真値と 一致（丸めは フラグ対象外・docstring の 意味論）
        lies += int((ex & (t.to(F32).double() != r.val.double())).sum())
    print(f"  {N:,} 件 × mul/add/div: NaN/Inf **{bad_naninf}**・フラグの嘘 **{lies}**"
          f"（意味論: 飽和/ε/0除算のみ フラグ・最近接丸めは 対象外）")

    print()
    print("=" * 76)
    print("② 配線表の 差し替え = 同じカーネルが 別の代数に（GPU 版 wiring_registry）")
    print("=" * 76)
    def ref_mult_M(x, y, OM, M):
        r = [0] * M
        for i in range(M):
            for j in range(M):
                r[i ^ j] += OM[i, j] * x[i] * y[j]
        return r
    for kind, M, name in [("cd", 2, "複素"), ("cd", 4, "四元数"), ("cd", 16, "セデニオン"),
                          ("cyclic", 8, "巡回畳み込みZ/8")]:
        T = wiring_tensor(kind, M, dev)
        bad = 0
        for _ in range(200):
            a = rng.integers(-9, 10, M).astype(np.float32)
            b = rng.integers(-9, 10, M).astype(np.float32)
            c = group_mul(T, Tot(torch.tensor(a, device=dev)), Tot(torch.tensor(b, device=dev)))
            got = [int(v) for v in c.val.cpu().numpy()]
            if kind == "cd":
                ref = ref_mult_M([int(x) for x in a], [int(x) for x in b], cd_omega(M), M)
            else:
                ref = [int(sum(a[i] * b[(k - i) % M] for i in range(M))) for k in range(M)]
            if got != ref: bad += 1
        print(f"  {name:<14} M={M:>2}: 違反 {bad}/200 {'✓' if bad == 0 else '×'}")

    print()
    print("=" * 76)
    print("③ スループット（5090・セデニオン積 バッチ）— 参考値")
    print("=" * 76)
    import time
    M = 16; T = wiring_tensor("cd", M, dev)
    for NB in (10_000, 1_000_000):
        a = Tot(torch.randn(NB, M, device=dev))
        b = Tot(torch.randn(NB, M, device=dev))
        group_mul(T, a, b)                                     # ウォームアップ
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            group_mul(T, a, b)
        if dev.type == "cuda": torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        print(f"  バッチ {NB:>9,}: {dt*1e3:7.2f} ms/回 = {NB/dt/1e6:8.1f} M sed積/s"
              f"（フラグ・無NaN 込み）")

    print()
    print("=" * 76)
    print("④ 入口の全域化 + 回帰（外部AI監査 2026-07-19 の 反例を 常設化）")
    print("=" * 76)
    t = Tot(torch.tensor([float('nan'), float('inf'), float('-inf'), 1e300, -1e300],
                         dtype=torch.float64, device=dev))
    ok_entry = (not torch.isnan(t.val).any()) and (not torch.isinf(t.val).any())
    print(f"  Tot([NaN,±Inf,±1e300]) → NaN/Inf 残留 {'なし ✓' if ok_entry else 'あり ×'}"
          f"（flag={t.flag.tolist()}）")
    zero = tot_mul(Tot(torch.zeros(1, device=dev)),
                   Tot(torch.tensor([1e300], dtype=torch.float64, device=dev)))
    zok = not torch.isnan(zero.val).any()
    print(f"  0 × Tot(1e300): val={zero.val.item():g}（旧版は NaN）{'✓' if zok else '×'}")
    ra = Tot(torch.tensor([MIN], device=dev)); ra.flag = torch.tensor([LE], dtype=torch.uint8, device=dev)
    rr = tot_add(ra, Tot(torch.tensor([-MIN], device=dev)))
    reg_ok = int(rr.flag.item()) == (GE | LE | SUNK)
    print(f"  (+MIN,LE)+(−MIN,=): flag={int(rr.flag.item())} = 境界なし+SUNK {'✓' if reg_ok else '×（旧版: LE=嘘）'}")
    assert ok_entry and zok and reg_ok

    print()
    print("=" * 76)
    print("⑤ フラグ代数の オラクル検査（外部AI監査の 盲点指摘に 応答）")
    print("   フラグ付き入力の 許容真値集合から 真値を 乱択し、出力フラグの 主張と 照合")
    print("=" * 76)
    K = 200_000
    rng5 = np.random.default_rng(11)
    def rand_flagged(K):
        """第3ラウンド監査の 盲点指摘を 反映: 表示0のフラグ付き(12%)・SUNK単独・
           GE|LE|SUNK・倍率は 10^6 まで・±MIN 近傍も 生成。"""
        mag = torch.tensor(10.0 ** rng5.uniform(-35, 20, K), device=dev)
        sgn = torch.tensor(rng5.choice([-1.0, 1.0], K), device=dev)
        val = (mag * sgn).to(F32)
        zero = torch.tensor(rng5.random(K) < 0.12, device=dev)
        val = torch.where(zero, torch.zeros_like(val), val)
        fl = torch.tensor(rng5.choice(
            [0, GE, LE, SUNK, GE | SUNK, LE | SUNK, GE | LE, GE | LE | SUNK], K).astype(np.uint8),
            device=dev)
        ge_, le_ = (fl & GE) > 0, (fl & LE) > 0
        u = torch.tensor(rng5.uniform(0, 1, K), device=dev)
        big = torch.tensor(10.0 ** rng5.uniform(0, 6, K), device=dev)   # 無制限側の 倍率
        m = torch.ones(K, dtype=torch.float64, device=dev)
        m = torch.where(ge_ & ~le_, 1 + big * u, m)              # GE: |真| ≥ |val|・上は 自由
        m = torch.where(le_ & ~ge_, u, m)                        # LE: |真| ∈ |val|·[0,1]
        m = torch.where(ge_ & le_, big * u, m)                   # 境界なし: 何でも
        def draw_true():
            """同一の (val, flag) から 許容真値を 独立に 引く（二証人 方式・第4ラウンド）。"""
            u2 = torch.tensor(rng5.uniform(0, 1, K), device=dev)
            big2 = torch.tensor(10.0 ** rng5.uniform(0, 6, K), device=dev)
            m2 = torch.ones(K, dtype=torch.float64, device=dev)
            m2 = torch.where(ge_ & ~le_, 1 + big2 * u2, m2)
            m2 = torch.where(le_ & ~ge_, u2, m2)
            m2 = torch.where(ge_ & le_, big2 * u2, m2)
            freemag = torch.tensor(10.0 ** rng5.uniform(-38, 20, K), device=dev)
            magt = torch.where(zero, torch.where(ge_, freemag, torch.zeros_like(freemag)),
                               val.double().abs() * m2)          # 表示0+GEビット ⟹ 真値自由
            sgn_unknown = ((fl & SUNK) > 0) | (zero & ge_)
            ts = torch.where(sgn_unknown,
                             torch.tensor(rng5.choice([-1.0, 1.0], K), device=dev).double(),
                             torch.sign(val).double())
            return magt * ts
        tt = Tot(val); tt.flag = fl
        return tt, draw_true(), draw_true()
    A2, ta, ta2 = rand_flagged(K); B2, tb, tb2 = rand_flagged(K)
    lies5 = 0
    def truth(name, x, y):
        return {"mul": x * y, "add": x + y,
                "div": torch.where(y == 0, torch.zeros_like(x),          # 真の分母0 ⟹ 規約 a/0=0
                                   x / torch.where(y == 0, torch.ones_like(y), y))}[name]
    for name, op in [("mul", tot_mul), ("add", tot_add), ("div", tot_div)]:
        r = op(A2, B2)
        t = truth(name, ta, tb); t2 = truth(name, ta2, tb2)
        ge = (r.flag & GE) > 0; le = (r.flag & LE) > 0; sk = (r.flag & SUNK) > 0
        vo = r.val.double(); slack = 2.0 ** -20                  # f32 丸め分の 猶予
        lies5 += int((ge & ~le & (t.abs() < vo.abs() * (1 - slack))).sum())
        lies5 += int((le & ~ge & (t.abs() > vo.abs() * (1 + slack))).sum())
        lies5 += int((~ge & ~le                                  # GE/LEなし = 大きさ厳密の 主張
                      & ((t.abs() - vo.abs()).abs() > vo.abs() * slack)).sum())
        lies5 += int(((~sk) & (vo != 0) & (t != 0)
                      & (torch.sign(t) != torch.sign(vo))).sum())  # 符号は 値が 運ぶ（SUNK 以外）
        # 第4ラウンドの 2契約: SUNKなし ⟹ ①表示0なら 真も0 ②二証人の 符号が 割れない
        lies5 += int(((~sk) & (vo == 0) & (t != 0)).sum())
        lies5 += int(((~sk) & (torch.sign(t) * torch.sign(t2) < 0)).sum())
        lies5 += int(torch.isnan(r.val).sum() + torch.isinf(r.val).sum())
    print(f"  {3*K:,} 件×二証人: 嘘 **{lies5}**（片側境界・大きさ厳密・符号・表示0・証人一致・無NaN）")
    assert lies5 == 0

    print()
    print("=" * 76)
    print("⑥ group_mul の オラクル検査（第2ラウンド SUNK 指摘 + パターン則の 健全性/保持率）")
    print("=" * 76)
    def rand_flagged_mat(KB, M):
        tt, tv, _ = rand_flagged(KB * M)
        m = Tot(tt.val.reshape(KB, M)); m.flag = tt.flag.reshape(KB, M)
        return m, tv.reshape(KB, M)
    def check6(label, kind, M, KB, mode):
        T = wiring_tensor(kind, M, dev)
        A, ta = rand_flagged_mat(KB, M); B, tb = rand_flagged_mat(KB, M)
        if mode == "sparse":                                    # 2成分だけ 非ゼロ（A/B 独立マスク）
            for X, txname in ((A, "a"), (B, "b")):
                keep = torch.zeros(KB, M, dtype=torch.bool, device=dev)
                keep.scatter_(1, torch.randint(0, M, (KB, 2), device=dev), True)
                X.val = torch.where(keep, X.val, torch.zeros_like(X.val))
                X.flag = torch.where(keep, X.flag, torch.zeros_like(X.flag))
                if txname == "a": ta = torch.where(keep, ta, torch.zeros_like(ta))
                else:             tb = torch.where(keep, tb, torch.zeros_like(tb))
        elif mode == "positive":                                # 全正・符号既知
            for X in (A, B):
                X.val = X.val.abs(); X.flag = X.flag & ~SUNK
            ta, tb = ta.abs(), tb.abs()
        r = group_mul(T, A, B)
        t = torch.einsum('kij,bi,bj->bk', T.double(), ta, tb)
        ge = (r.flag & GE) > 0; le = (r.flag & LE) > 0; sk = (r.flag & SUNK) > 0
        vo = r.val.double(); slack = 2.0 ** -20
        lies = int((ge & ~le & (t.abs() < vo.abs() * (1 - slack))).sum())
        lies += int((le & ~ge & (t.abs() > vo.abs() * (1 + slack))).sum())
        lies += int(((~sk) & (vo != 0) & (t != 0)
                     & (torch.sign(t) != torch.sign(vo))).sum())
        lies += int(torch.isnan(r.val).sum() + torch.isinf(r.val).sum())
        # 保持率: フラグを含む 入力行のうち、何かを 主張できた 出力成分の 割合
        frow = (((A.flag | B.flag)) > 0).any(dim=-1, keepdim=True).expand(r.flag.shape)
        claims = ((ge ^ le) | ~sk) & frow
        ret = float(claims.sum()) / max(int(frow.sum()), 1)
        print(f"  {label:<22} {KB:,}行: 嘘 **{lies}**・主張保持率 {ret:5.1%}")
        return lies
    bad6 = check6("四元数・密±乱数", "cd", 4, 20_000, "dense")
    bad6 += check6("セデニオン・疎(2成分)", "cd", 16, 5_000, "sparse")
    bad6 += check6("巡回Z/8・全正(畳込)", "cyclic", 8, 10_000, "positive")
    # 監査の 原反例（M=1・単項 SUNK）: 大きさは 保持・符号だけ 不明、が 正解
    T1 = wiring_tensor("cd", 1, dev)
    xs = Tot(torch.tensor([[2.0]], device=dev)); xs.flag = torch.tensor([[SUNK]], dtype=torch.uint8, device=dev)
    r1 = group_mul(T1, xs, Tot(torch.tensor([[3.0]], device=dev)))
    reg6 = int(r1.flag.item()) == SUNK and float(r1.val.item()) == 6.0
    print(f"  原反例 (2,SUNK)×(3,=): val={r1.val.item():g} flag={int(r1.flag.item())} "
          f"= 大きさ厳密+符号不明 {'✓' if reg6 else '×'}")
    # 第3ラウンド回帰: 「表示0 だが 真値自由」の 項を 死項にしない
    z = Tot(torch.tensor([[0.0]], device=dev)); z.flag = torch.tensor([[GE | LE | SUNK]], dtype=torch.uint8, device=dev)
    rA = group_mul(T1, z, Tot(torch.tensor([[3.0]], device=dev)))
    regA = int(rA.flag.item()) == (GE | LE | SUNK)
    print(f"  反例1 (0,GE|LE|SUNK)×(3,=): flag={int(rA.flag.item())} = 境界なし+SUNK {'✓' if regA else '×（旧: 0=嘘）'}")
    T2c = wiring_tensor("cyclic", 2, dev)
    a2 = Tot(torch.tensor([[2.0, 0.0]], device=dev)); a2.flag = torch.tensor([[0, GE | LE | SUNK]], dtype=torch.uint8, device=dev)
    rB = group_mul(T2c, a2, Tot(torch.tensor([[3.0, 1.0]], device=dev)))
    regB = all(int(f) == (GE | LE | SUNK) for f in rB.flag[0])
    print(f"  反例2 巡回M=2 隠れ項: flag={rB.flag[0].tolist()} = 両成分 境界なし+SUNK {'✓' if regB else '×（旧: [0,0]=嘘）'}")
    # SUNK単独 加算の 回帰（第3ラウンドを 受けた 自前発見）: |±2±3| ∈ {1,5} ⟹ 大きさ厳密は 嘘
    su = Tot(torch.tensor([2.0], device=dev)); su.flag = torch.tensor([SUNK], dtype=torch.uint8, device=dev)
    sv = Tot(torch.tensor([3.0], device=dev)); sv.flag = torch.tensor([SUNK], dtype=torch.uint8, device=dev)
    rC = tot_add(su, sv)
    regC = int(rC.flag.item()) == (GE | LE | SUNK)
    print(f"  (2,SUNK)+(3,SUNK): flag={int(rC.flag.item())} = 境界なし+SUNK {'✓' if regC else '×（旧: SUNK=大きさ厳密の嘘）'}")
    # 第4ラウンド回帰: 危険な0 の 意味論を スカラー演算にも（group_mul とだけ 一致していた）
    dz = Tot(torch.tensor([0.0], device=dev)); dz.flag = torch.tensor([GE], dtype=torch.uint8, device=dev)
    e3 = Tot(torch.tensor([3.0], device=dev))
    rD = tot_mul(dz, e3)
    regD = (int(rD.flag.item()) & SUNK) > 0
    print(f"  (0,GE)×(3,=): flag={int(rD.flag.item())} SUNKあり {'✓' if regD else '×（旧: GE のみ=符号の嘘）'}")
    e1 = Tot(torch.tensor([1.0], device=dev))
    rE = tot_div(e1, dz)
    regE = int(rE.flag.item()) == (GE | LE | SUNK)
    print(f"  (1,=)/(0,GE): flag={int(rE.flag.item())} = 境界なし+SUNK {'✓' if regE else '×（旧: SUNKなし）'}")
    dz7 = Tot(torch.tensor([0.0], device=dev)); dz7.flag = torch.tensor([GE | LE | SUNK], dtype=torch.uint8, device=dev)
    rF = tot_div(e1, dz7)
    regF = int(rF.flag.item()) == (GE | LE | SUNK)
    print(f"  (1,=)/(0,GE|LE|SUNK): flag={int(rF.flag.item())} {'✓' if regF else '×'}")
    # 本当の0（符号なし・向きは±MINが運ぶ）は 吸収: (0,=)×(3,GE) → (0, フラグなし)
    e3g = Tot(torch.tensor([3.0], device=dev)); e3g.flag = torch.tensor([GE], dtype=torch.uint8, device=dev)
    rG = tot_mul(Tot(torch.tensor([0.0], device=dev)), e3g)
    regG = int(rG.flag.item()) == 0 and float(rG.val.item()) == 0.0
    print(f"  (0,=)×(3,GE): flag={int(rG.flag.item())} = 厳密な真の0 {'✓' if regG else '×'}")
    assert bad6 == 0 and reg6 and regA and regB and regC and regD and regE and regF and regG

    print()
    print("GPU 版: 全域算術（無NaN・フラグ）+ 配線表差し替え + 飽和は最後に1回、が torch で 動く。")


if __name__ == "__main__":
    self_test()
