# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
TotArith — a scalar total-arithmetic Number for Julia.

  `TotNum <: Real`: value + a flag that names when the true value left the machine's
  representable range, WITH direction.  Overflow → ±MAX + GE (|true| ≥ |val|);
  underflow → ±MIN + LE (|true| ≤ |val|); a/0 = 0; NaN/Inf are never produced.

  Because it subtypes `Real` and overloads `Base.:+ - * /` etc., *existing generic
  Julia code runs on it unchanged* — the flag flows through any library that is written
  against `Number`/`Real` (ODE solvers, linear algebra, ...).  That is the whole point.
"""
module ScalarTot

export TotNum, GE, LE, SUNK, isflagged, flag_of, MAXF, MINF

const GE   = 0x01
const LE   = 0x02
const SUNK = 0x04
const MAXF = floatmax(Float64)
const MINF = floatmin(Float64)

struct TotNum <: Real
    val::Float64
    flag::UInt8
end
TotNum(x::Real) = _entry(Float64(x))          # entry totalization at construction
TotNum(x::TotNum) = x

flag_of(a::TotNum) = a.flag
isflagged(a::TotNum) = a.flag != 0x00
Base.Float64(a::TotNum) = a.val
Base.Float32(a::TotNum) = Float32(a.val)
Base.float(a::TotNum) = a.val
Base.AbstractFloat(a::TotNum) = a.val
(::Type{T})(a::TotNum) where {T<:AbstractFloat} = T(a.val)
Base.float(::Type{TotNum}) = TotNum
Base.big(a::TotNum) = big(a.val)

# ---- totalize a raw Float64 into (val, flag). Never NaN/Inf. ----
@inline function _sat(raw::Float64)
    isnan(raw) && return TotNum(0.0, GE | LE | SUNK)
    s = sign(raw); a = abs(raw)
    if a > MAXF || isinf(raw); return TotNum(s * MAXF, GE); end
    if a > 0 && a < MINF;      return TotNum(s * MINF, LE); end
    return TotNum(raw, 0x00)
end
@inline _entry(x::Float64) = _sat(x)

@inline function _addflag(fa, fb, va, vb)
    fin = fa | fb
    (fin == 0x00) && return 0x00
    same = sign(va) * sign(vb) > 0
    known = (fin & SUNK) == 0x00
    (known && same) ? fin : (GE | LE | SUNK)   # cancellation-safe
end
@inline function _mulflag(fa, fb)
    ga = fa & GE; la = (fa >> 1) & 0x01
    gb = fb & GE; lb = (fb >> 1) & 0x01
    ge = ((ga | gb) & ~(la | lb)) & 0x01
    le = ((la | lb) & ~(ga | gb)) & 0x01
    nb = ((ga | gb) & (la | lb)) & 0x01
    (ge * GE) | (le * LE) | (nb * (GE | LE)) | ((fa | fb) & SUNK)
end

# ---- the operator overloads: THIS is the bridge to the whole ecosystem ----
function Base.:+(a::TotNum, b::TotNum)
    r = _sat(a.val + b.val)
    TotNum(r.val, r.flag | _addflag(a.flag, b.flag, a.val, b.val))
end
function Base.:-(a::TotNum, b::TotNum)
    r = _sat(a.val - b.val)
    TotNum(r.val, r.flag | _addflag(a.flag, b.flag, a.val, -b.val))
end
Base.:-(a::TotNum) = TotNum(-a.val, a.flag)
function Base.:*(a::TotNum, b::TotNum)
    r = _sat(a.val * b.val)
    tz = (a.val == 0 && (a.flag & GE) == 0) || (b.val == 0 && (b.flag & GE) == 0)
    tz ? TotNum(0.0, 0x00) : TotNum(r.val, r.flag | _mulflag(a.flag, b.flag))
end
function Base.:/(a::TotNum, b::TotNum)
    bz = b.val == 0
    raw = bz ? 0.0 : a.val / b.val               # a/0 = 0
    r = _sat(raw)
    fin = a.flag | b.flag
    nb = (fin & (GE | LE)) > 0
    dz = (a.val == 0 && (a.flag & GE) > 0) || (b.val == 0 && (b.flag & GE) > 0)
    f = r.flag | (nb ? (GE | LE) : 0x00) | (fin & SUNK) | (dz ? SUNK : 0x00)
    TotNum(r.val, f)
end

# ---- the glue that lets generic Number/Real code accept TotNum ----
Base.promote_rule(::Type{TotNum}, ::Type{<:Real}) = TotNum
Base.convert(::Type{TotNum}, x::Real) = TotNum(x)
Base.zero(::Type{TotNum}) = TotNum(0.0, 0x00)
Base.one(::Type{TotNum})  = TotNum(1.0, 0x00)
Base.zero(::TotNum) = zero(TotNum)
Base.one(::TotNum)  = one(TotNum)
# comparisons + basics the ecosystem calls
Base.:<(a::TotNum, b::TotNum) = a.val < b.val
Base.:<=(a::TotNum, b::TotNum) = a.val <= b.val
Base.:(==)(a::TotNum, b::TotNum) = a.val == b.val
Base.isless(a::TotNum, b::TotNum) = a.val < b.val
Base.abs(a::TotNum) = TotNum(abs(a.val), a.flag)
Base.sign(a::TotNum) = TotNum(sign(a.val), (a.flag & SUNK))
# ---- transcendental flag algebra --------------------------------------------------------
# GE/LE are ABSOLUTE-VALUE bounds. They survive a function ONLY when the function is
# monotone on the admissible set AND the direction survives |·| — that must be PROVEN per
# function, not assumed. Principle (external audit 2026-07-20, five confirmed lies):
# when it cannot be proven, drop to GE|LE|SUNK (no bound, sign untrusted) — an honest
# "I know nothing" beats a stale direction.
const CPLX = 0x08   # "真の結果が real 欄に収まらない(可能性を含む)" — √-1型は確定・符号不明入力は可能性
const NOB  = GE | LE                              # no bound in either direction

# 符号を信用できない入力(SUNK / 危険な0)は、定義域が符号に敏感な関数では
# 「真値が負→実数の外」の可能性を持つ ⇒ CPLX まで立てるのが健全 (意味論オラクルで強制)
@inline _sign_untrusted(a::TotNum) = (a.flag & SUNK) != 0 || (a.val == 0 && (a.flag & GE) != 0)

function Base.sqrt(a::TotNum)
    if a.val < 0                                            # 定義域外: 複素へ、と名指し
        # 符号不明なら 真値が正で実結果もあり得る → 古い方向ビットを残さず NOB に落とす
        return (a.flag & SUNK) != 0 ? TotNum(0.0, NOB | SUNK | CPLX) : TotNum(0.0, a.flag | CPLX)
    end
    _sign_untrusted(a) && return TotNum(sqrt(a.val), NOB | SUNK | CPLX)
    # sign trusted, val ≥ 0: |·|^½ is monotone in magnitude → direction bits survive
    r = _sat(sqrt(a.val))
    TotNum(r.val, r.flag | (a.flag & (GE | LE)))
end
function Base.:^(a::TotNum, b::TotNum)
    x, y = a.val, b.val
    inflag = a.flag | b.flag
    # 指数=0 の 二種を 分ける（"0を予約語に"の 帰結）: 本物の0 だけが 空の積=1。
    # 指数が 整数だが Int64 に 収まらない(1e300 等) → 実数冪 exp(y·log x) の 経路へ
    # (旧版は isinteger(y) で 整数扱い→Int64(y) が InexactError を 投げた・全域監査で 発覚)
    if y == 0
        if b.flag == 0x00                          # 指数が **本物の0** → 空の積 → 1 (0^0 も 1)
            return TotNum(1.0, a.flag & SUNK)      # (底の 符号不明だけは 伝播・大きさは 1 で 確定)
        else                                       # 指数が **潰れた≈0(±MIN)** = 微小な非ゼロ → 空の積でない
            if x == 0 && a.flag == 0x00            #   0^(微小): 符号+なら0/−なら∞ → 割れる
                return TotNum(0.0, GE | LE | SUNK) #   確定できない → 境界なし+符号不明
            else                                   #   有限底: a^(微小) ≈ 1 (連続)
                return TotNum(1.0, inflag)
            end
        end
    end
    if x < 0 && !isinteger(y)                     # (負)^(非整数) = 実数の範囲外 → 型が違う
        # 底の符号が不明なら 真値が正=実結果もあり得る → 方向ビットを主張しない
        return (a.flag & SUNK) != 0 ? TotNum(0.0, NOB | SUNK | CPLX) :
                                      TotNum(0.0, inflag | CPLX)
    end
    if x == 0 && y < 0                            # 0^負 = 1/(0^|y|) = 1/0 = 0 (全域規約と整合)
        _sign_untrusted(a) || return TotNum(0.0, 0x00)      # 本物の0(や true=0 のLE) → 厳密に0
        return TotNum(0.0, NOB | SUNK | (isinteger(y) ? 0x00 : CPLX))
    end
    # 負の底の 符号: 指数の 偶奇で 決まる。y が Int64 外(1e300 等)なら 実質 偶数扱いで 安全
    s = (x < 0 && abs(y) < 9e18 && isodd(round(Int, y))) ? -1.0 : 1.0
    r = _sat(s * abs(x)^y)                         # 溢れ→±MAX·GE / 潰れ→±MIN·LE を _sat が担当
    if b.flag != 0x00
        return TotNum(r.val, r.flag | NOB | SUNK)  # 指数が不確か: 方向を主張できない
    end
    if !isinteger(y) && _sign_untrusted(a)
        return TotNum(r.val, r.flag | NOB | SUNK | CPLX)   # 底が負かも×非整数冪 → 複素の可能性
    end
    TotNum(r.val, r.flag | _powflag(a.flag, y))
end
# |result| = |x|^y is monotone in |x| ⇒ the |·|-bound direction survives — but y < 0 is a
# reciprocal, so GE and LE SWAP (audit counterexample ⑤: (2,GE)^-1 claimed 0.5⟦≥⟧ while the
# truth allows 4⁻¹ = 0.25 ≤ 0.5). Sign: even integer power ⇒ certain (+, SUNK dropped);
# odd ⇒ follows the base (SUNK kept); non-integer with untrusted sign ⇒ possibly complex
# ⇒ NOB|SUNK (the 3-bit vocabulary's honest floor).
@inline function _powflag(fbase::UInt8, y::Float64)
    dir = fbase & (GE | LE)
    if y < 0
        dir = dir == GE ? LE : (dir == LE ? GE : dir)
    end
    (fbase & SUNK) == 0x00 && return dir
    if isinteger(y) && abs(y) < 9e18
        return iseven(round(Int, y)) ? dir : (dir | SUNK)
    end
    NOB | SUNK
end
function Base.:^(a::TotNum, n::Integer)
    n == 0 && return TotNum(1.0, 0x00)             # x^0 = 1 exact (0^0 = 1: 空の積)
    if a.val == 0 && n < 0                          # 0^(−n) = 1/0 = 0 (Moore-Penrose と整合)
        return a.flag == 0x00 ? TotNum(0.0, 0x00) : TotNum(0.0, NOB | SUNK)
    end
    r = _sat(a.val^n)
    TotNum(r.val, r.flag | _powflag(a.flag, Float64(n)))
end
Base.literal_pow(::typeof(^), a::TotNum, ::Val{N}) where {N} = a^N
function Base.exp(a::TotNum)
    r = _sat(exp(a.val))
    f = a.flag
    f == 0x00 && return r
    # exp > 0 always ⇒ output sign is CERTAIN (SUNK never propagates out); but the
    # |·|-bound direction flips with the input's sign: |true|≥|val| with val<0 means
    # true ≤ val ⇒ exp(true) ≤ exp(val) ⇒ LE, not GE. (audit counterexample ③)
    (f & SUNK) != 0 && return TotNum(r.val, r.flag | NOB)
    dir = f & (GE | LE)
    dir == NOB && return TotNum(r.val, r.flag | NOB)
    if a.val == 0
        out = dir == LE ? 0x00 : NOB          # (0,LE) ⇒ true=0 ⇒ exact 1 ; (0,GE) ⇒ anything
    elseif a.val > 0
        out = dir                              # positive: monotone, direction survives
    else
        out = dir == GE ? LE : GE              # negative: direction flips
    end
    TotNum(r.val, r.flag | out)
end
function Base.log(a::TotNum)
    if a.val < 0                                            # 実数範囲外: 複素へ
        return (a.flag & SUNK) != 0 ? TotNum(0.0, NOB | SUNK | CPLX) : TotNum(0.0, a.flag | CPLX)
    end
    if a.val == 0
        # 本物の0: log0 = −∞ → −MAX+GE ; 危険な0(GE付き): 真値不明(負なら複素) → NOB|SUNK|CPLX
        return (a.flag & GE) != 0 ? TotNum(0.0, NOB | SUNK | CPLX) : TotNum(-MAXF, GE)
    end
    r = _sat(log(a.val))
    f = a.flag
    f == 0x00 && return r
    (f & SUNK) != 0 && return TotNum(r.val, r.flag | NOB | SUNK | CPLX)  # 真値が負なら複素
    dir = f & (GE | LE)
    # log crosses sign at 1: the |·|-bound direction survives only when the admissible set
    # stays on one side of 1 — provable in exactly two cells (audit principle):
    out = if dir == GE && a.val > 1
        GE                                     # true ≥ val > 1: log>0 grows → GE
    elseif dir == LE && a.val < 1
        GE                                     # 0 < true ≤ val < 1: |log| grows toward 0⁺ → GE
    else
        NOB | SUNK                             # admissible set may cross 1: sign+bound unknown
    end
    TotNum(r.val, r.flag | out)
end
# periodic: an input magnitude-bound says (almost) NOTHING about the output — the
# admissible set can cross any number of periods (audit counterexample ④: sin(π/2,GE)
# claimed +1⟦≥⟧ while the truth allows sin(3π/2) = −1). Flagged input ⇒ NOB|SUNK.
Base.sin(a::TotNum) = a.flag == 0x00 ? TotNum(sin(a.val), 0x00) : TotNum(sin(a.val), NOB | SUNK)
Base.cos(a::TotNum) = a.flag == 0x00 ? TotNum(cos(a.val), 0x00) : TotNum(cos(a.val), NOB | SUNK)
Base.inv(a::TotNum) = one(TotNum) / a
Base.:*(a::TotNum, b::Bool) = b ? a : zero(TotNum)   # solvers multiply by Bool masks
Base.:*(b::Bool, a::TotNum) = b ? a : zero(TotNum)
Base.nextfloat(a::TotNum) = TotNum(nextfloat(a.val), a.flag)
Base.prevfloat(a::TotNum) = TotNum(prevfloat(a.val), a.flag)
Base.eps(::Type{TotNum}) = TotNum(eps(Float64), 0x00)
Base.eps(a::TotNum) = TotNum(eps(a.val), 0x00)         # 適応刻みが 値に対して呼ぶ
Base.oneunit(::Type{TotNum}) = one(TotNum)
Base.abs2(a::TotNum) = a * a                           # ソルバの ノルムが 使う
Base.isapprox(a::TotNum, b::TotNum; kw...) = isapprox(a.val, b.val; kw...)
Base.max(a::TotNum, b::TotNum) = a.val >= b.val ? a : b
Base.min(a::TotNum, b::TotNum) = a.val <= b.val ? a : b
Base.round(a::TotNum) = TotNum(round(a.val), a.flag)
Base.:*(a::TotNum, b::Integer) = a * TotNum(b)
Base.:/(a::TotNum, b::Integer) = a / TotNum(b)
Base.:^(a::TotNum, b::AbstractFloat) = a ^ TotNum(b)
Base.typemax(::Type{TotNum}) = TotNum(MAXF, GE)
Base.typemin(::Type{TotNum}) = TotNum(-MAXF, GE)
Base.isnan(::TotNum) = false                     # never — by construction
Base.isinf(::TotNum) = false
Base.isfinite(::TotNum) = true
Base.show(io::IO, a::TotNum) =
    print(io, a.flag == 0 ? string(a.val) :
              string(a.val, "⟦", (a.flag & GE)>0 ? "≥" : "", (a.flag & LE)>0 ? "≤" : "",
                     (a.flag & SUNK)>0 ? "±" : "", (a.flag & CPLX)>0 ? "ℂ" : "", "⟧"))

end # module
