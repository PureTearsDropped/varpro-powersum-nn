"""全域化された算術（利用者の規約）。

    **溢れ  → ±MAX**   符号を保つ。Inf にしない
    **潰れ  → ±MIN**   符号を保つ。**0 にしない**
    **a/0  → 0**       本物の 0 で割ったときだけ
    **0×0  → 0**
    **x^0  → 1**       全ての x で（空の積）

要点は速度でも精度でもなく、**`0` を予約語にすること**である。IEEE では `0` が
「本当に 0」と「小さすぎて表現できなかった」を兼務している。兼務している限り
`a/0 = 0`（＝「測れなかった点は無いことにする」）は自分の入力を偽造する:
潰れた本物の計算が `0` に化け、そこで `a/0` が誤発動し、**巨大な値が 0 になる**
（向きが真逆の嘘であって、有界な嘘ではない）。

±MIN が入って初めて `0` は「本物の 0」しか意味しなくなり、`a/0 = 0` が
一つの意味を持てる。**ゼロ除算の全域化には、アンダーフローの全域化が要る。**

副作用として、演算が NaN も Inf も返さなくなる（`total_arith` の自己検査で
324/324 通り確認済み）。したがって `isfinite` の見張りは書く理由が無くなる。

**適用範囲の制限（重要。§7.7.2）**:

    `zero_ok=(a==0 or b==0)` は **隠れた仮定** を持つ:

        **a·b = 0  ⟹  a = 0 または b = 0**   （＝ 零因子が無い）

    実数では真。八元数までは真。**セデニオン（16次元）で初めて偽になる**:

        (e1 + e10)(e4 − e15) = 0     ← 両方 0 でないのに、厳密に 0

    このとき本モジュールは **厳密な 0 を「潰れ」と誤診して ±MIN を返す**（6成分すべてで確認）。

    ⟹ **本モジュールは、零因子の無い代数でのみ正しい。**

    **直し方は在る（§7.7.3。未実装）**: `x` を「左から掛ける」行列 `L(x)` にすると
    **零因子 ⟺ L(x) が特異**であり、

        ~~zero_ok = 掛ける前の**値**が 0 か~~
        **zero_ok = b が L(a) の零空間に居るか**   ← 6/6（スカラー 3/6）、実数では後方互換

    が正しい判定になる（＝「積が 0 なのは **構造上か、桁が足りないか**」）。
    **値には書いていないが、行列には書いてある。** 費用（L は O(M²)、pinv は O(M³)）と、
    閾値なしの実装は未検討。
"""
import numpy as np

MAX = np.finfo(np.float64).max          # 1.7976931348623157e+308
MIN = np.nextafter(0.0, 1.0)            # 4.9406564584124654e-324（最小の正の非正規数）


def saturate(v, zero_ok):
    """溢れ→±MAX、潰れ→±MIN、NaN→0。`zero_ok` が真の要素だけ 0 のままにする。

    zero_ok は「その要素は **本物の 0** であるべき」という呼び出し側の主張である
    （例: 0 × 5.0 の結果、0^w の結果）。それ以外の 0 はアンダーフローなので ±MIN。
    """
    v = np.asarray(v, float)
    zero_ok = np.broadcast_to(np.asarray(zero_ok, bool), v.shape)
    with np.errstate(all="ignore"):
        sign = np.where(np.signbit(v), -1.0, 1.0)               # -0.0 の符号も拾う
        v = np.where(np.isnan(v), 0.0, v)
        v = np.where(np.isposinf(v), MAX, np.where(np.isneginf(v), -MAX, v))
        underflow = (v == 0.0) & ~zero_ok
        v = np.where(underflow, sign * MIN, v)
        return np.clip(v, -MAX, MAX)


def total_power(x, w):
    """全域化された |x|^w。

        w = 0            → 1        全ての x で（空の積）
        x = 0, w ≠ 0     → 0        w>0 は正しい。w<0 は **嘘だが有界**
        それ以外          → |x|^w    溢れたら ±MAX、潰れたら +MIN。**0 にはしない**
    """
    x = np.asarray(x, float)
    if w == 0.0:
        return np.ones_like(x)
    is_zero = np.abs(x) == 0.0
    with np.errstate(all="ignore"):
        v = np.power(np.abs(x), float(w))
    v = saturate(v, zero_ok=is_zero)            # |x|^w は非負。潰れは +MIN へ
    return np.where(is_zero, 0.0, v)


def total_div(a, b):
    """a/0 = 0（本物の 0 のときだけ）。それ以外は溢れ→±MAX、潰れ→±MIN。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(all="ignore"):
        v = np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=(b != 0.0))
    v = saturate(v, zero_ok=(a == 0.0))
    return np.where(b == 0.0, 0.0, v)           # **b が本物の 0 → 0**


def total_mul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(all="ignore"):
        return saturate(a * b, zero_ok=(a == 0.0) | (b == 0.0))


def self_check(verbose=False):
    """NaN も Inf も出ないこと、0 が本物の 0 からしか出ないことを総当たりで確認。"""
    vals = np.array([0.0, 1.0, -1.0, MAX, -MAX, MIN, -MIN, 1e308, 1e-308])
    bad_nonfinite = bad_fake_zero = total = 0
    for a in vals:
        for b in vals:
            for op, fn in (("*", total_mul), ("/", total_div)):
                total += 1
                r = float(fn(a, b))
                if not np.isfinite(r):
                    bad_nonfinite += 1
                if r == 0.0 and a != 0.0 and b != 0.0:      # 0 が偽造された
                    bad_fake_zero += 1
    for x in vals:
        for w in (-3.0, -0.5, 0.0, 0.5, 3.0, 300.0, -300.0):
            total += 1
            r = float(total_power(x, w))
            if not np.isfinite(r):
                bad_nonfinite += 1
            if r == 0.0 and x != 0.0:
                bad_fake_zero += 1
    if verbose:
        print(f"  total_arith 自己検査: {total} 通り")
        print(f"    Inf か NaN を返した   : **{bad_nonfinite}/{total}**")
        print(f"    0 を偽造した（潰れ由来）: **{bad_fake_zero}/{total}**")
    return bad_nonfinite, bad_fake_zero, total


if __name__ == "__main__":
    self_check(verbose=True)
