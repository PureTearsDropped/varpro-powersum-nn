# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
NestedSeries — the M/N/O layers as FREELY COMPOSABLE registries (Julia twin of
nested_series.py, generalized).

  Everything is an `Alg`: a bilinear algebra (dim, unit, structure table).  The three
  layers are three registries over that one interface:

    N layer (cells)      : cd_alg(M) (ℝ/ℂ/ℍ/𝕆/sedenion), cyclic_alg(M) (group ℤ/M),
                           matn_alg(n) (real n×n matrices)  — the wiring tables
    M layer (combinators): mat_over(alg, N) (N×N matrix of cells),
                           tensor(A, B)     (A ⊗ B — two wiring tables multiplied)
                           — each RETURNS a new Alg, so they nest recursively:
                           mat_over(tensor(cd_alg(4), cd_alg(2)), 2) just works.
    O layer (tapes)      : TAPES — series coefficients (exp, sin, cos, sinh, cosh, …)
                           + a declared bracket (:left / :right) for building powers.

  Nothing about a combination is assumed: `assoc_defect(alg)` MEASURES whether the
  composed algebra is associative, and `nlog` (inverse ⇒ candidate) verifies its answer
  with the safe forward exp — unverifiable ⇒ INEXACT flag, never a silent lie.
  Total: elements carry (coeffs, flag); NaN→0+SING, overflow→±MAX+OVER at every step.

  Measured laws this module lets you reproduce (self_test):
    · associativity survives composition iff every ingredient is associative
      (cd(≤4), cyclic, matn, and their tensors/matrices — but one octonion cell
      infects the whole tower)
    · exp∘log = id verifies (1e-15) exactly on the associative combinations and
      breaks structurally (≈1e-3) on the non-associative ones — same code, same tape
    · brackets :left / :right agree on scalars (power-associativity), split for
      matrices of non-associative cells (the "many exps")
"""
module NestedSeries

import LinearAlgebra                       # stdlib: I と pinv(自己テストの二証人)のみ使用

export Alg, cd_alg, cyclic_alg, matn_alg, grassmann_alg, clifford_alg,
       mat_over, tensor, jordan, lie, commutator, ALGS, alg, list_algs,
       Lmat_alg, Rmat_alg, nsolve_left, nsolve_right,
       nconj, nconj_div_left, nconj_div_right, nnorm_div, nnormalize,
       nleft_action, nright_action,
       Nel, nel, coeffs, flagof, tmul, tadd,
       TAPES, series, nexp, nsin, ncos, nsinh, ncosh, nexp_ss, nlog, ninv,
       OPS, nop, list_ops, binom_tape,
       assoc_defect, powerassoc_defect, commut_defect, SING, OVER, INEXACT

const SING    = 0x01
const OVER    = 0x04
const INEXACT = 0x08
const MAXF = floatmax(Float64)

# ================================================================ the one interface
"""A bilinear algebra: `dim`, `unit` (the 1), and the structure table
   `tab[i][j] :: Vector` = eᵢ·eⱼ expanded in the basis. The table IS the wiring."""
struct Alg
    name::String
    dim::Int
    unit::Vector{Float64}
    tab::Vector{Vector{Vector{Float64}}}      # tab[i][j] = basis product eᵢ eⱼ
end
Base.show(io::IO, A::Alg) = print(io, A.name, "(dim ", A.dim, ")")

function _from_mul(name, d, unit, mul)        # extract the wiring table once
    E = [Float64.(1:d .== i) for i in 1:d]
    Alg(name, d, unit, [[mul(E[i], E[j]) for j in 1:d] for i in 1:d])
end

"raw bilinear product through the wiring table (dense loops; clarity over speed)"
function rawmul(A::Alg, x::Vector{Float64}, y::Vector{Float64})
    r = zeros(A.dim)
    for i in 1:A.dim
        xi = x[i]; xi == 0.0 && continue
        ti = A.tab[i]
        for j in 1:A.dim
            yj = y[j]; yj == 0.0 && continue
            r .+= (xi * yj) .* ti[j]
        end
    end
    r
end

# ================================================================ N layer: cell registry
function _cdconj(x); n = length(x); n == 1 ? copy(x) : vcat(x[1], -x[2:end]); end
function _cdprod(x, y)
    n = length(x); n == 1 && return x .* y
    h = n ÷ 2
    a, b, c, d = x[1:h], x[h+1:end], y[1:h], y[h+1:end]
    vcat(_cdprod(a, c) .- _cdprod(_cdconj(d), b), _cdprod(d, a) .+ _cdprod(b, _cdconj(c)))
end
"Cayley–Dickson algebra of dim M: ℝ(1) ℂ(2) ℍ(4) 𝕆(8) sedenion(16) …"
cd_alg(M::Int) = _from_mul("cd$M", M, Float64.(1:M .== 1), _cdprod)

"group algebra of ℤ/M: eᵢ·eⱼ = e_{(i+j) mod M} — commutative AND associative"
cyclic_alg(M::Int) = _from_mul("cyc$M", M, Float64.(1:M .== 1),
    (x, y) -> begin
        r = zeros(M)
        for i in 0:M-1, j in 0:M-1
            r[mod(i + j, M) + 1] += x[i+1] * y[j+1]
        end
        r
    end)

"real n×n matrices as a dim-n² algebra (column-major vec) — associative, with zero divisors"
matn_alg(n::Int) = _from_mul("mat$n", n * n, vec(Matrix{Float64}(I0(n))),
    (x, y) -> vec(reshape(x, n, n) * reshape(y, n, n)))
I0(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

"sign of reordering basis blades: (−1)^#{(i,j): i∈A, j∈B, i>j} (bitmask blades)"
function _reorder_sign(A::Int, B::Int)
    cnt = 0
    for i in 0:62
        (A >> i) & 1 == 1 || continue
        cnt += count_ones(B & ((1 << i) - 1))     # pairs (i∈A, j∈B, j<i) = transpositions
    end
    iseven(cnt) ? 1.0 : -1.0
end

"""Grassmann (exterior) algebra Λℝⁿ, dim 2ⁿ: eᵢeⱼ = −eⱼeᵢ, eᵢ² = 0 — every generator is
   NILPOTENT, so series TERMINATE (exp is exactly a polynomial).  grassmann_alg(1) is the
   dual numbers a+bε: f(a+ε) = f(a)+f′(a)ε — forward-mode automatic differentiation falls
   out of the shelf as an algebra."""
grassmann_alg(n::Int) = _from_mul("Λ$n", 1 << n, Float64.(1:(1 << n) .== 1),
    (x, y) -> begin
        D = 1 << n; r = zeros(D)
        for a in 0:D-1
            xa = x[a+1]; xa == 0.0 && continue
            for b in 0:D-1
                yb = y[b+1]; yb == 0.0 && continue
                a & b == 0 || continue                        # overlap ⇒ eᵢ² = 0
                r[(a ⊻ b) + 1] += _reorder_sign(a, b) * xa * yb
            end
        end
        r
    end)

"""Clifford algebra Cl(n,0), dim 2ⁿ: the geometric product — eᵢeⱼ = −eⱼeᵢ but eᵢ² = +1
   (same wiring as Grassmann with the overlap surviving instead of dying).  The Julia
   mirror of the hardware repo's `_clifford_omega`."""
clifford_alg(n::Int) = _from_mul("Cl$n", 1 << n, Float64.(1:(1 << n) .== 1),
    (x, y) -> begin
        D = 1 << n; r = zeros(D)
        for a in 0:D-1
            xa = x[a+1]; xa == 0.0 && continue
            for b in 0:D-1
                yb = y[b+1]; yb == 0.0 && continue
                r[(a ⊻ b) + 1] += _reorder_sign(a, b) * xa * yb
            end
        end
        r
    end)

# ================================================================ M layer: combinators
"""N×N matrix over any Alg — a new Alg of dim N²·cell.dim (block index (r,c,k)).
   The matrix product's summation order is fixed; whether the RESULT is associative
   depends on the cell (measure with assoc_defect, don't assume)."""
function mat_over(cell::Alg, N::Int)
    d = cell.dim; D = N * N * d
    at(r, c, k) = ((c - 1) * N + (r - 1)) * d + k          # column-major blocks
    unit = zeros(D); for i in 1:N, k in 1:d; unit[at(i, i, k)] = cell.unit[k]; end
    mul = (x, y) -> begin
        r = zeros(D)
        xb = (i, j) -> x[at(i, j, 1):at(i, j, d)]
        yb = (i, j) -> y[at(i, j, 1):at(i, j, d)]
        for i in 1:N, j in 1:N
            acc = zeros(d)
            for m in 1:N
                acc .+= rawmul(cell, xb(i, m), yb(m, j))
            end
            r[at(i, j, 1):at(i, j, d)] = acc
        end
        r
    end
    _from_mul("mat$(N)⟨$(cell.name)⟩", D, unit, mul)
end

"tensor product A ⊗ B — two wiring tables multiplied: (eₐ⊗f_b)(e_c⊗f_d) = (eₐe_c)⊗(f_bf_d)"
function tensor(A::Alg, B::Alg)
    dA, dB = A.dim, B.dim; D = dA * dB
    at(a, b) = (a - 1) * dB + b
    unit = zeros(D)
    for a in 1:dA, b in 1:dB; unit[at(a, b)] = A.unit[a] * B.unit[b]; end
    mul = (x, y) -> begin
        r = zeros(D)
        for a in 1:dA, b in 1:dB
            xab = x[at(a, b)]; xab == 0.0 && continue
            for c in 1:dA, d in 1:dB
                ycd = y[at(c, d)]; ycd == 0.0 && continue
                sA = A.tab[a][c]; sB = B.tab[b][d]
                for p in 1:dA
                    sA[p] == 0.0 && continue
                    for q in 1:dB
                        r[at(p, q)] += xab * ycd * sA[p] * sB[q]
                    end
                end
            end
        end
        r
    end
    _from_mul("$(A.name)⊗$(B.name)", D, unit, mul)
end

# ================================================================ total elements + ops
"""symmetrized (Jordan) product a∘b = (ab+ba)/2 — commutative by construction, but
   associativity is generally LOST (measure it). This is the 'symmetrized exp' member
   of the exp family made into a combinator. Measured role: a commutative-but-non-
   associative tensor partner does NOT preserve power-associativity — commutativity
   alone is not enough, the partner must be commutative AND associative."""
function jordan(A::Alg)
    _from_mul("sym⟨$(A.name)⟩", A.dim, copy(A.unit),
              (x, y) -> (rawmul(A, x, y) .+ rawmul(A, y, x)) ./ 2)
end

"""antisymmetrized product ½[a,b] = (ab−ba)/2 — jordan's sibling: the ORDER-ONLY half.
   Every product splits EXACTLY as  ab = a∘b + ½[a,b]  (order-forgetting + order-carrying);
   commutativity is precisely "the lie half vanishes".  Measured ladder of what the
   commutator machinery can repair (see self_test):
     · 2-variable BCH  exp(a)exp(b) = exp(a+b+½[a,b]+1/12[a,[a,b]]+1/12[b,[b,a]]+…)
       repairs at s⁴-scaling up to the OCTONIONS (Artin: any 2-generated subalgebra is
       associative) and breaks to s³ at the sedenions (alternativity lost).
     · 3-variable Jacobi [[a,b],c]+[[b,c],a]+[[c,a],b]=0 breaks already at the octonions
       (the commutator algebra is Malcev, not Lie)."""
function lie(A::Alg)
    _from_mul("lie⟨$(A.name)⟩", A.dim, zeros(A.dim),        # no unit: [1,x]=0 kills it
              (x, y) -> (rawmul(A, x, y) .- rawmul(A, y, x)) ./ 2)
end
"the commutator [a,b] = ab − ba on Nel — the order information itself"
commutator(A::Alg, x, y) = tadd(tmul(A, x, y), tscale(tmul(A, y, x), -1.0))

"""ALGS — the algebra preset shelf, OPS's twin on the N/M side.  Famous algebras by name,
   each built from the registered cells and combinators (a preset IS a composition —
   :dualquat is literally tensor(grassmann_alg(1), cd_alg(4))).  `alg(:name)` grabs one;
   `list_algs()` prints each preset's MEASURED id-card (assoc / pow-assoc / commut) —
   the shelf never asserts a property it hasn't measured."""
const ALGS = Dict{Symbol,Function}(
    :real       => () -> cd_alg(1),
    :complex    => () -> cd_alg(2),
    :quaternion => () -> cd_alg(4),
    :octonion   => () -> cd_alg(8),
    :sedenion   => () -> cd_alg(16),
    :split      => () -> cyclic_alg(2),               # j² = +1: zero divisors at dim 2
    :dual       => () -> grassmann_alg(1),            # ε² = 0: forward-mode AD
    :grassmann2 => () -> grassmann_alg(2),            # fermions: everything nilpotent
    :cl2        => () -> clifford_alg(2),             # geometric product (≅ M₂ℝ)
    :cl3        => () -> clifford_alg(3),             # Pauli algebra
    :dualquat   => () -> tensor(grassmann_alg(1), cd_alg(4)),  # rigid-body pose (drones)
    :biquat     => () -> tensor(cd_alg(2), cd_alg(4)),         # complexified quaternions
    :m4real     => () -> tensor(cd_alg(4), cd_alg(4)),         # ℍ⊗ℍ ≅ M₄ℝ (measured earlier)
)
alg(name::Symbol) = ALGS[name]()

"print the shelf with each algebra's measured id-card — properties observed, not declared"
function list_algs()
    println(rpad("preset", 12), rpad("realizes", 22), rpad("dim", 5),
            rpad("assoc", 7), rpad("pow-assoc", 11), "commut")
    for nm in sort(collect(keys(ALGS)))
        A = ALGS[nm]()
        g = _lcg()
        ad = assoc_defect(A; rng = g); pa = powerassoc_defect(A; rng = g)
        cd_ = commut_defect(A; rng = g)
        println(rpad(string(nm), 12), rpad(A.name, 22), rpad(string(A.dim), 5),
                rpad(ad < 1e-9 ? "✓" : "✗", 7), rpad(pa < 1e-9 ? "✓" : "✗", 11),
                cd_ < 1e-9 ? "✓" : "✗")
    end
end

"element of an Alg: coefficients + flag; totalized at every step (never NaN/Inf)"
struct Nel
    c::Vector{Float64}
    flag::UInt8
end
nel(A::Alg, c::AbstractVector) = _tot(Float64.(collect(c)), 0x00)
nel(A::Alg) = Nel(copy(A.unit), 0x00)                       # the 1
coeffs(x::Nel) = x.c
flagof(x::Nel) = x.flag
function _tot(c::Vector{Float64}, f::UInt8)
    for i in eachindex(c)
        v = c[i]
        if isnan(v); c[i] = 0.0; f |= SING
        elseif !isfinite(v) || abs(v) > MAXF; c[i] = sign(v) * MAXF; f |= OVER
        end
    end
    Nel(c, f)
end
tmul(A::Alg, x::Nel, y::Nel) = _tot(rawmul(A, x.c, y.c), x.flag | y.flag)
tadd(x::Nel, y::Nel) = _tot(x.c .+ y.c, x.flag | y.flag)
tscale(x::Nel, s::Float64) = _tot(x.c .* s, x.flag)

# ================================================================ O layer: tape registry
const TAPES = Dict{Symbol,Function}(
    :exp  => k -> 1.0 / factorial(big(k)),
    :sin  => k -> iseven(k) ? 0.0 : Float64((-1)^((k - 1) ÷ 2) / factorial(big(k))),
    :cos  => k -> isodd(k)  ? 0.0 : Float64((-1)^(k ÷ 2) / factorial(big(k))),
    :sinh => k -> iseven(k) ? 0.0 : 1.0 / factorial(big(k)),
    :cosh => k -> isodd(k)  ? 0.0 : 1.0 / factorial(big(k)),
)

"""Σ c_k x^k on ANY Alg, powers built by the DECLARED bracket
   (:left → x^k = x^{k-1}·x, :right → x·x^{k-1}). One skeleton, many tapes."""
function series(A::Alg, x::Nel, tape; order::Int = 20, bracket::Symbol = :left)
    c = tape isa Symbol ? TAPES[tape] : tape
    acc = tscale(nel(A), Float64(c(0)))
    P = nel(A)
    for k in 1:order
        P = bracket === :left ? tmul(A, P, x) : tmul(A, x, P)
        ck = Float64(c(k))
        ck != 0.0 && (acc = tadd(acc, tscale(P, ck)))
    end
    acc
end
nexp(A, x; kw...)  = series(A, x, :exp;  kw...)
nsin(A, x; kw...)  = series(A, x, :sin;  order = 21, kw...)
ncos(A, x; kw...)  = series(A, x, :cos;  kw...)
nsinh(A, x; kw...) = series(A, x, :sinh; order = 21, kw...)
ncosh(A, x; kw...) = series(A, x, :cosh; kw...)

"exp by scaling-and-squaring — a DIFFERENT cell connection; agreement with nexp is measured"
function nexp_ss(A::Alg, x::Nel; order::Int = 12, s::Int = 3, bracket::Symbol = :left)
    acc = series(A, tscale(x, 1.0 / 2^s), :exp; order, bracket)
    for _ in 1:s; acc = tmul(A, acc, acc); end
    acc
end

"""log = inverse ⇒ CANDIDATE: series log(1+X) (X = x − 1, needs ‖X‖ small), then verified
   by the safe forward exp; unverified ⇒ INEXACT — a candidate, never a silent lie."""
function nlog(A::Alg, x::Nel; order::Int = 30, verify_order::Int = 20)
    X = tadd(x, tscale(nel(A), -1.0))
    y = series(A, X, k -> k == 0 ? 0.0 : (-1.0)^(k + 1) / k; order)
    resid = maximum(abs.(coeffs(nexp(A, y; order = verify_order)).- x.c))
    resid < 1e-6 ? (y, resid) : (Nel(y.c, y.flag | INEXACT), resid)
end

"""1/x WITHOUT a divider: the all-ones tape Σ u^k = (1−u)⁻¹ with u = 1 − x (converges for
   ‖u‖ < 1), verified TWO-SIDED (x·y ≈ 1 AND y·x ≈ 1 — left and right inverse can differ
   in a non-commutative algebra, so both are checked).  Inverse ⇒ candidate: a zero divisor
   (or any x outside the basin) fails verification and is flagged INEXACT — the series
   diverges honestly instead of returning a lie.  This is division rebuilt from the same
   cells as everything else: one more coefficient tape on the one skeleton."""
function ninv(A::Alg, x::Nel; order::Int = 60)
    u = tadd(nel(A), tscale(x, -1.0))
    y = series(A, u, k -> 1.0; order)
    resid = max(maximum(abs.(coeffs(tmul(A, x, y)) .- A.unit)),
                maximum(abs.(coeffs(tmul(A, y, x)) .- A.unit)))
    resid < 1e-6 ? (y, resid) : (Nel(y.c, y.flag | INEXACT), resid)
end

# ================================================================ operator presets
"binomial tape for (1+u)^p — the coefficient c_k = C(p,k), built iteratively"
binom_tape(p) = k -> begin
    c = 1.0
    for i in 1:k; c *= (p - i + 1) / i; end
    c
end

"""OPS — the operator preset shelf.  Each entry is one operation as data:
     kind    :forward (safe for every input) | :candidate (verified, else INEXACT)
     tape    the coefficient series (the O layer)
     shift   false: series in x | true: series in u = x − 1 (log/inv/roots live near 1)
     verify  for candidates: (A, x, y) -> residual of the DEFINING identity, brackets
             declared inside (e.g. cbrt checks (y·y)·y).
   Adding an operation = adding one entry.  `nop(A, name, x)` runs any of them on any Alg."""
const OPS = Dict{Symbol,NamedTuple}(
    :exp   => (kind = :forward,   tape = TAPES[:exp],  shift = false, order = 20, verify = nothing),
    :sin   => (kind = :forward,   tape = TAPES[:sin],  shift = false, order = 21, verify = nothing),
    :cos   => (kind = :forward,   tape = TAPES[:cos],  shift = false, order = 20, verify = nothing),
    :sinh  => (kind = :forward,   tape = TAPES[:sinh], shift = false, order = 21, verify = nothing),
    :cosh  => (kind = :forward,   tape = TAPES[:cosh], shift = false, order = 20, verify = nothing),
    :atan  => (kind = :forward,   tape = k -> isodd(k) ? (-1.0)^((k - 1) ÷ 2) / k : 0.0,
               shift = false, order = 41, verify = nothing),
    :log   => (kind = :candidate, tape = k -> k == 0 ? 0.0 : (-1.0)^(k + 1) / k,
               shift = true, order = 30,
               verify = (A, x, y) -> maximum(abs.(coeffs(nexp(A, y)) .- x.c))),
    :inv   => (kind = :candidate, tape = k -> (-1.0)^k, shift = true, order = 60,
               # Σ(x−1)^k(−1)^k = Σ(1−x)^k — the geometric series, in shift bookkeeping
               verify = (A, x, y) -> max(maximum(abs.(coeffs(tmul(A, x, y)) .- A.unit)),
                                         maximum(abs.(coeffs(tmul(A, y, x)) .- A.unit)))),
    :sqrt  => (kind = :candidate, tape = binom_tape(0.5), shift = true, order = 40,
               verify = (A, x, y) -> maximum(abs.(coeffs(tmul(A, y, y)) .- x.c))),
    :cbrt  => (kind = :candidate, tape = binom_tape(1 / 3), shift = true, order = 40,
               verify = (A, x, y) -> maximum(abs.(coeffs(tmul(A, tmul(A, y, y), y)) .- x.c))),
)
"""run a preset by name on any Alg: `nop(A, :sqrt, x)`.  Forward presets are total for
   every input; candidates verify their defining identity and flag INEXACT on failure —
   same honesty for every operator, uniformly."""
function nop(A::Alg, name::Symbol, x::Nel; order::Union{Int,Nothing} = nothing,
             bracket::Symbol = :left)
    op = OPS[name]
    ord = order === nothing ? op.order : order
    arg = op.shift ? tadd(x, tscale(nel(A), -1.0)) : x
    y = series(A, arg, op.tape; order = ord, bracket)
    op.kind === :forward && return y
    resid = op.verify(A, x, y)
    resid < 1e-6 ? y : Nel(y.c, y.flag | INEXACT)
end

"print the preset shelf: name, kind, and what the candidate verification checks"
function list_ops()
    for (nm, op) in sort(collect(OPS); by = first)
        println(rpad(string(nm), 7), op.kind === :forward ? "forward (total, no flag needed)" :
                "candidate (verified vs defining identity, else INEXACT)")
    end
end

# ================================================================ solve: 方程式を解く除算
# 「a/0 = 0 は Moore–Penrose の 1×1」の フルランク完成: solve_left は 同じ定理の dim×dim。
# 除算の 家系(外部レビュー 2026-07-21 の 区別を 実装):
#   代数式  x·ā/|a|²  … 常に 計算できるが、非結合(セデニオン)では ay=x の 解とは 限らない
#   方程式の解 L_a⁺x  … ay=x の 最小ノルム最小二乗解。零因子でも 定義される
# 実装は 乗算だけの Ben-Israel 反復(判断・除算・ピボットなし = 固定配線可 = newton_recip の
# 行列版)。検算は 二層: 前向き残差 a·y≈x (厳密解) / 正規方程式残差 (最小二乗)。
# フラグ: 厳密解→クリーン / 解なし(最小二乗のみ)→SING / 未収束→INEXACT — 解けたフリをしない。

"left-multiplication matrix: (a·y)_k = Σ_i a_i tab[i][j][k] y_j — 構築は配線(積なし)"
function Lmat_alg(A::Alg, a::AbstractVector)
    L = zeros(A.dim, A.dim)
    for i in 1:A.dim
        ai = a[i]; ai == 0.0 && continue
        ti = A.tab[i]
        for j in 1:A.dim, k in 1:A.dim
            L[k, j] += ai * ti[j][k]
        end
    end
    L
end

"right-multiplication matrix: (y·a)_k = Σ_j a_j tab[i][j][k] y_i"
function Rmat_alg(A::Alg, a::AbstractVector)
    R = zeros(A.dim, A.dim)
    for j in 1:A.dim
        aj = a[j]; aj == 0.0 && continue
        for i in 1:A.dim
            tij = A.tab[i][j]
            for k in 1:A.dim
                R[k, i] += aj * tij[k]
            end
        end
    end
    R
end

"""乗算だけの擬似逆(Ben-Israel): X₀=Mᵀ·2^{-s} → X(2I−MX)。特異でも A⁺ に二次収束。
   割り算ゼロ: 反復は 乗算と減算だけ。唯一のスケール 1/‖M‖₁‖M‖∞ は「上界なら何でもよい」ので
   次の 2 のベキに 切り上げ ⟹ 指数の引き算 = 底 2 の付け替え = ハードでは ゲート 0 個
   (gate_series の 2^{-s} と 同じ手筋)。solve は 端から端まで セルの数珠つなぎになる。"""
function _pinv_mul(Mx::Matrix{Float64}, K::Int)
    n1 = maximum(sum(abs, Mx; dims = 1)); ninf = maximum(sum(abs, Mx; dims = 2))
    d = n1 * ninf
    d == 0.0 && return zeros(size(Mx, 2), size(Mx, 1))   # a=0: L=0 ⟹ L⁺=0 (a/0=0 と同型)
    X = Mx' .* exp2(-ceil(log2(d)))                       # 2^{-s}: 指数シフトのみ(除算不使用)
    for _ in 1:K
        X = X * (2 * LinearAlgebra.I - Mx * X)
    end
    X
end

function _solve_via(Mmat, mulfn, A::Alg, a::Nel, x::Nel, K::Int, tol::Float64)
    Op = Mmat(A, a.c)
    y = _tot(_pinv_mul(Op, K) * x.c, a.flag | x.flag)
    r1 = maximum(abs.(coeffs(mulfn(y)) .- x.c))              # 前向き検算: 方程式が解けたか
    r2 = maximum(abs.(Op' * (coeffs(mulfn(y)) .- x.c)))      # 正規方程式: 最小二乗の検算
    f = r1 < tol ? y.flag :
        (r2 < tol ? (y.flag | SING) : (y.flag | SING | INEXACT))
    (Nel(y.c, f), r1, r2)
end

"solve a·y = x: 最小ノルム最小二乗解 L_a⁺x。厳密解→クリーン / 解なし→SING / 未収束→INEXACT"
nsolve_left(A::Alg, a::Nel, x::Nel; K::Int = 30, tol::Float64 = 1e-8) =
    _solve_via(Lmat_alg, y -> tmul(A, a, y), A, a, x, K, tol)

"solve y·a = x: R_a⁺x (非可換なので 左と 一般に 別解)"
nsolve_right(A::Alg, a::Nel, x::Nel; K::Int = 30, tol::Float64 = 1e-8) =
    _solve_via(Rmat_alg, y -> tmul(A, y, a), A, a, x, K, tol)

# ---------------------------------------------------------------- exp の家族(残り2人)
# exp の 5 分類の 完備: 左結合/右結合 = series(bracket)・対称化 = nexp(jordan(A),·)・
# 左作用/右作用 = exp(t·L_a)·x₀ / x₀·exp(t·R_a)。除算の家族と 同じハブ L_a/R_a を 使う:
# exp(L_a) は 流れ(ẋ=a·x の解)・L_a⁺ は 除算 — 1 つの 行列の 2 つの 顔。
# 恒等式(自己テストで検証): 単位元に 当てると 左作用 = 左結合 exp・右作用 = 右結合 exp。

function _action_series(Op::Matrix{Float64}, x0::Nel, t::Float64, order::Int, inflag::UInt8)
    acc = copy(x0.c); term = copy(x0.c)
    for k in 1:order
        term = (t / k) .* (Op * term)                    # (tL)ᵏ/k!·x₀ — 行列×ベクトルだけ
        acc .+= term
    end
    _tot(acc, inflag)
end

"左作用 exp: x(t) = exp(t·L_a)·x₀ = 線形 ODE ẋ = a·x の解。乗算だけの級数"
nleft_action(A::Alg, a::Nel, x0::Nel, t::Float64; order::Int = 24) =
    _action_series(Lmat_alg(A, a.c), x0, t, order, a.flag | x0.flag)

"右作用 exp: x(t) = exp(t·R_a)·x₀ = ẋ = x·a の解"
nright_action(A::Alg, a::Nel, x0::Nel, t::Float64; order::Int = 24) =
    _action_series(Rmat_alg(A, a.c), x0, t, order, a.flag | x0.flag)

# ---------------------------------------------------------------- 除算の家族(残り3人)
# 外部レビュー(2026-07-21)の 5 分類を 棚に 完備: ①conj-div ②solve_left ③solve_right
# ④norm_div ⑤normalize (+ ninv=幾何級数テープ)。①は「常に計算できる 代数式」であって
# 「方程式の解」とは 限らない — 解になるのは 合成代数 dim 1,2,4,8 (Hurwitz, self-test 実測)。
# だから ①は 検算し、解でないときは INEXACT を 立てる: 形式的な式であることを 値が 自分で 語る。

"共役 (e0成分以外を反転 — CD 族の標準対合。他の代数では検算フラグが守る)"
nconj(A::Alg, a::Nel) = Nel(vcat(a.c[1], -a.c[2:end]), a.flag)

function _conj_div(A::Alg, a::Nel, x::Nel, left::Bool, tol::Float64)
    n2 = sum(abs2, a.c)
    n2 == 0.0 && return (Nel(zeros(A.dim), a.flag | x.flag), Inf)   # a=0: a/0=0 と同型
    cj = nconj(A, a)
    y = tscale(left ? tmul(A, cj, x) : tmul(A, x, cj), 1.0 / n2)
    r = maximum(abs.(coeffs(left ? tmul(A, a, y) : tmul(A, y, a)) .- x.c))
    (r < tol ? y : Nel(y.c, y.flag | INEXACT), r)                    # 解でない→形式的な式と名指し
end

"conj-div 左: (ā·x)/|a|² — a·y=x の解になれば clean・ならなければ INEXACT (Hurwitz が門番)"
nconj_div_left(A::Alg, a::Nel, x::Nel; tol::Float64 = 1e-8) = _conj_div(A, a, x, true, tol)

"conj-div 右: (x·ā)/|a|² — y·a=x 用"
nconj_div_right(A::Alg, a::Nel, x::Nel; tol::Float64 = 1e-8) = _conj_div(A, a, x, false, tol)

"ノルム比 ‖x‖/‖a‖ (実数・方向情報なし)。‖a‖=0 → 0 (スカラーの a/0=0)"
function nnorm_div(x::Nel, a::Nel)
    na = sqrt(sum(abs2, a.c))
    na == 0.0 ? 0.0 : sqrt(sum(abs2, x.c)) / na
end

"正規化 a/‖a‖ (実スカラー除算のみ=左右・括弧の曖昧さなし)。a=0 → 0"
function nnormalize(A::Alg, a::Nel)
    na = sqrt(sum(abs2, a.c))
    na == 0.0 ? Nel(zeros(A.dim), a.flag) : Nel(a.c ./ na, a.flag)
end

# ================================================================ measure, don't assume
"max |(xy)z − x(yz)| over random triples — the associativity of the COMPOSED algebra"
function assoc_defect(A::Alg; trials::Int = 4, rng = nothing)
    rnd = rng === nothing ? _lcg() : rng
    worst = 0.0
    for _ in 1:trials
        x, y, z = (Nel(0.3 .* rand_vec(rnd, A.dim), 0x00) for _ in 1:3)
        l = tmul(A, tmul(A, x, y), z); r = tmul(A, x, tmul(A, y, z))
        worst = max(worst, maximum(abs.(l.c .- r.c)))
    end
    worst
end

"""max |(xx)x − x(xx)| — POWER-associativity, the true gate for single-element series:
   Cayley–Dickson scalars keep it even when non-associative (octonion, sedenion), so
   exp∘log verifies there; matrix/tensor composites can LOSE it — measure, don't assume."""
function powerassoc_defect(A::Alg; trials::Int = 4, rng = nothing)
    rnd = rng === nothing ? _lcg() : rng
    worst = 0.0
    for _ in 1:trials
        x = Nel(0.3 .* rand_vec(rnd, A.dim), 0x00)
        x2 = tmul(A, x, x)
        worst = max(worst, maximum(abs.(tmul(A, x2, x).c .- tmul(A, x, x2).c)))
    end
    worst
end
"max |xy − yx| — commutativity of the composed algebra (the third probe)"
function commut_defect(A::Alg; trials::Int = 4, rng = nothing)
    rnd = rng === nothing ? _lcg() : rng
    worst = 0.0
    for _ in 1:trials
        x, y = (Nel(0.3 .* rand_vec(rnd, A.dim), 0x00) for _ in 1:2)
        worst = max(worst, maximum(abs.(tmul(A, x, y).c .- tmul(A, y, x).c)))
    end
    worst
end

mutable struct _LCG; s::UInt64; end
_lcg() = _LCG(0x9E3779B97F4A7C15)
function rand_vec(g::_LCG, n)
    v = zeros(n)
    for i in 1:n
        g.s = g.s * 6364136223846793005 + 1442695040888963407
        v[i] = (Float64(g.s >> 11) / 2.0^53) * 2 - 1
    end
    v
end

# ================================================================ self-test
function self_test()
    println("NestedSeries — every combination measured, none assumed")
    combos = [
        cd_alg(2), cd_alg(4), cd_alg(8), cd_alg(16), cyclic_alg(6), matn_alg(2),
        mat_over(cd_alg(4), 2), mat_over(cd_alg(16), 2),
        tensor(cd_alg(4), cd_alg(4)), tensor(cd_alg(8), cd_alg(2)),
        mat_over(tensor(cd_alg(4), cd_alg(2)), 2),          # free recursion: mat(H⊗C)
    ]
    println(rpad("algebra", 26), rpad("dim", 6), rpad("assoc", 9), rpad("pow-assoc", 11),
            rpad("exp(0)=1", 10), rpad("exp∘log", 12), "verdict")
    for A in combos
        g = _lcg()
        ad = assoc_defect(A; rng = g)
        pa = powerassoc_defect(A; rng = g)
        e0ok = maximum(abs.(coeffs(nexp(A, Nel(zeros(A.dim), 0x00))) .- A.unit)) < 1e-12
        x = Nel(0.25 .* rand_vec(g, A.dim), 0x00)
        xnear = tadd(nel(A), tscale(x, 0.5))
        _, resid = nlog(A, xnear)
        verdict = resid < 1e-6 ? "✓ inverse pair" : "✗ INEXACT (structural)"
        println(rpad(A.name, 26), rpad(string(A.dim), 6),
                rpad(ad < 1e-9 ? "✓" : "✗", 9),
                rpad(pa < 1e-9 ? "✓" : "✗ $(round(pa, sigdigits=2))", 11),
                rpad(e0ok ? "✓" : "✗", 10),
                rpad(string(round(resid, sigdigits = 2)), 12), verdict)
        @assert e0ok
        # measured law: exp∘log verifies iff POWER-associativity holds (not full
        # associativity — octonion/sedenion scalars are the counterexample that
        # falsified the naive "assoc ⟺ verify" version of this assertion)
        @assert (pa < 1e-9) == (resid < 1e-6) "pow-assoc/verify mismatch on $(A.name)"
    end
    # brackets: agree on scalar cells, split for matrices of non-associative cells
    g = _lcg()
    x16 = Nel(0.3 .* rand_vec(g, 16), 0x00)
    dscalar = maximum(abs.(coeffs(nexp(cd_alg(16), x16)) .-
                           coeffs(nexp(cd_alg(16), x16; bracket = :right))))
    Am = mat_over(cd_alg(16), 2)
    xm = Nel(0.15 .* rand_vec(g, Am.dim), 0x00)
    dmat = maximum(abs.(coeffs(nexp(Am, xm)) .- coeffs(nexp(Am, xm; bracket = :right))))
    dss  = maximum(abs.(coeffs(nexp(Am, xm)) .- coeffs(nexp_ss(Am, xm))))
    println("brackets — scalar cd16 left vs right: ", round(dscalar, sigdigits = 2),
            " (agree)   mat2⟨cd16⟩ left vs right: ", round(dmat, sigdigits = 2),
            "  vs sqring: ", round(dss, sigdigits = 2), " (distinct exps)")
    @assert dscalar < 1e-9 && dmat > 1e-6
    # totality: NaN/huge input crashes nothing, names everything
    bad = nel(cd_alg(16), [NaN; fill(1e308, 15)])
    r = nexp(cd_alg(16), bad)
    @assert flagof(bad) & SING != 0 && all(isfinite, coeffs(r))
    println("totality: NaN/1e308 input → flags ", string(flagof(bad), base = 2),
            ", exp stays finite ✓")
    # ninv: division rebuilt as a tape — verified two-sided, INEXACT on zero divisors
    A16 = cd_alg(16); g2 = _lcg()
    xr = tadd(nel(A16), tscale(Nel(0.3 .* rand_vec(g2, 16), 0x00), 1.0))
    yinv, r1 = ninv(A16, xr)
    @assert r1 < 1e-6 && (flagof(yinv) & INEXACT) == 0
    zd = zeros(16); zd[4] = 1.0; zd[11] = 1.0                 # 1−x = e3+e10 zero divisor
    ybad, r2 = ninv(A16, tadd(nel(A16), tscale(nel(A16, zd), -1.0)))
    @assert (flagof(ybad) & INEXACT) != 0
    println("ninv: (1/x)·x = x·(1/x) = 1 at ", round(r1, sigdigits = 2),
            " ✓ ; zero-divisor → INEXACT ✓ (division as a tape, no divider)")
    # measured tensor law: a non-associative base keeps power-associativity under ⊗
    # ONLY when the partner is commutative AND associative — either alone fails.
    # (jordan(cd8) is the pincer: commutative ✓, associative ✗ → still loses it.)
    for (partner, keeps) in ((cyclic_alg(3), true), (cd_alg(4), false), (jordan(cd_alg(8)), false))
        T = tensor(cd_alg(8), partner); gt = _lcg()
        pa = powerassoc_defect(T; rng = gt)
        @assert (pa < 1e-9) == keeps "tensor law violated on $(T.name)"
        xn = tadd(nel(T), tscale(Nel(0.25 .* rand_vec(gt, T.dim), 0x00), 0.5))
        _, res = nlog(T, xn)
        @assert (pa < 1e-9) == (res < 1e-6) "pow-assoc/verify mismatch on $(T.name)"
    end
    println("tensor law: ⊗-partner must be commutative AND associative to preserve",
            " power-associativity (jordan pincer: commutative alone fails) ✓")
    # order machinery: exact split ab = a∘b + ½[a,b]; Jacobi and BCH gates measured
    for M in (4, 16)
        Ao = cd_alg(M); go = _lcg()
        a = Nel(0.4 .* rand_vec(go, M), 0x00); b = Nel(0.4 .* rand_vec(go, M), 0x00)
        Aj = jordan(Ao); Al = lie(Ao)
        recon = tadd(_tot(rawmul(Aj, a.c, b.c), 0x00), _tot(rawmul(Al, a.c, b.c), 0x00))
        @assert maximum(abs.(recon.c .- tmul(Ao, a, b).c)) < 1e-12
    end
    jac(A, x, y, z) = tadd(tadd(commutator(A, commutator(A, x, y), z),
                                commutator(A, commutator(A, y, z), x)),
                           commutator(A, commutator(A, z, x), y))
    jd = Dict{Int,Float64}()
    for M in (4, 8, 16)
        Ao = cd_alg(M); go = _lcg()
        x, y, z = (Nel(0.4 .* rand_vec(go, M), 0x00) for _ in 1:3)
        jd[M] = maximum(abs.(jac(Ao, x, y, z).c))
    end
    @assert jd[4] < 1e-12 && jd[8] > 1e-3 && jd[16] > 1e-3
    println("order split ab = a∘b + ½[a,b] exact ✓ ; Jacobi: cd4 ✓ Lie, cd8/cd16 ✗ (Malcev)")
    # BCH repair gate by scaling exponent: s⁴ (repaired) through octonions — Artin's
    # theorem measured — s³ (structural) at sedenions
    ratios = Dict{Int,Float64}()
    for M in (4, 8, 16)
        Ao = cd_alg(M); go = _lcg()
        ba = rand_vec(go, M); bb = rand_vec(go, M)
        r = Float64[]
        for s in (0.2, 0.1)
            a = Nel(s .* ba, 0x00); b = Nel(s .* bb, 0x00)
            lhs = tmul(Ao, nexp(Ao, a), nexp(Ao, b))
            zc = tadd(tadd(a, b), tscale(commutator(Ao, a, b), 0.5))
            zc = tadd(zc, tadd(tscale(commutator(Ao, a, commutator(Ao, a, b)), 1 / 12),
                               tscale(commutator(Ao, b, commutator(Ao, b, a)), 1 / 12)))
            push!(r, maximum(abs.(coeffs(lhs) .- coeffs(nexp(Ao, zc)))))
        end
        ratios[M] = r[1] / r[2]
    end
    @assert ratios[4] > 12 && ratios[8] > 12 && ratios[16] < 10
    println("BCH gate: cd4 ", round(ratios[4], sigdigits = 3), " / cd8 ",
            round(ratios[8], sigdigits = 3), " ≈ s⁴ repaired (Artin measured) ; cd16 ",
            round(ratios[16], sigdigits = 3), " ≈ s³ structural break ✓")
    # operator preset shelf: one gateway, uniform honesty
    Ap = cd_alg(16); gp = _lcg()
    xp = Nel(0.3 .* rand_vec(gp, 16), 0x00)
    for (nm, f) in ((:exp, nexp), (:sin, nsin), (:cos, ncos), (:sinh, nsinh), (:cosh, ncosh))
        @assert maximum(abs.(coeffs(nop(Ap, nm, xp)) .- coeffs(f(Ap, xp)))) < 1e-12
    end
    xn = tadd(nel(Ap), tscale(xp, 0.5))                       # near 1: roots/log/inv converge
    ys = nop(Ap, :sqrt, xn)
    @assert (flagof(ys) & INEXACT) == 0
    @assert maximum(abs.(coeffs(tmul(Ap, ys, ys)) .- xn.c)) < 1e-6
    yc = nop(Ap, :cbrt, xn)
    @assert (flagof(yc) & INEXACT) == 0                       # (y·y)·y bracket declared in verify
    A1 = cd_alg(1)                                            # reals: atan preset vs Base.atan
    @assert abs(coeffs(nop(A1, :atan, nel(A1, [0.5])))[1] - atan(0.5)) < 1e-9
    zsq = nop(mat_over(cd_alg(16), 2), :sqrt,
              tadd(nel(mat_over(cd_alg(16), 2)),
                   tscale(Nel(0.2 .* rand_vec(gp, 64), 0x00), 0.5)))
    println("preset shelf: exp/sin/cos/sinh/cosh ≡ named fns ✓ ; √,∛ verified on cd16 ✓ ; ",
            "atan(0.5) matches ℝ ✓ ; √ on mat2⟨cd16⟩ (no pow-assoc): ",
            (flagof(zsq) & INEXACT) == 0 ? "verifies (2-factor identity!)" : "INEXACT (measured)")
    # algebra preset shelf: id-cards measured live
    println("--- ALGS shelf (id-cards measured, not declared) ---")
    list_algs()
    # dual numbers = forward-mode AD: f(a+ε) = f(a) + f′(a)ε — derivatives for free
    D = alg(:dual); a0 = 1.2
    xd = nel(D, [a0, 1.0])                                   # a + ε
    for (nm, f, fp) in ((:exp, exp, exp), (:sin, sin, cos),
                        (:sqrt, sqrt, t -> 1 / (2 * sqrt(t))), (:inv, t -> 1 / t, t -> -1 / t^2))
        y = nop(D, nm, xd)
        @assert (flagof(y) & INEXACT) == 0
        @assert abs(coeffs(y)[1] - f(a0)) < 1e-9 && abs(coeffs(y)[2] - fp(a0)) < 1e-9
    end
    println("dual (=Λ1): f(a+ε)=f(a)+f′(a)ε for exp/sin/√/inv ✓ — AD falls out of the shelf")
    # dual quaternions (drone pose): associative composite → whole shelf verifies on it
    DQ = alg(:dualquat); gq = _lcg()
    @assert assoc_defect(DQ; rng = gq) < 1e-9
    xq = tadd(nel(DQ), tscale(Nel(0.3 .* rand_vec(gq, DQ.dim), 0x00), 0.5))
    _, rq = nlog(DQ, xq)
    @assert rq < 1e-6
    println("dualquat (=Λ1⊗ℍ, rigid-body pose): associative ✓, exp∘log ",
            round(rq, sigdigits = 2), " ✓ — the drone algebra straight off the shelf")
    # nilpotency: Λ2 series TERMINATE — exp of a pure blade is exact at tiny order
    G2 = alg(:grassmann2)
    xg = nel(G2, [0.0, 0.7, 0.4, 0.0])
    @assert maximum(abs.(coeffs(nexp(G2, xg; order = 3)) .- coeffs(nexp(G2, xg; order = 30)))) < 1e-15
    println("Λ2: nilpotent ⇒ exp terminates (order 3 ≡ order 30 exactly) ✓")
    # solve: 方程式を解く除算 (a/0=0 のフルランク完成)
    println("--- solve: L⁺/R⁺ (乗算だけの Ben-Israel・零因子対応・二層検算) ---")
    A16s = cd_alg(16); gs = _lcg()
    a_r = Nel(rand_vec(gs, 16), 0x00); x_r = Nel(rand_vec(gs, 16), 0x00)
    yL, r1, r2 = nsolve_left(A16s, a_r, x_r)
    @assert r1 < 1e-8 && flagof(yL) == 0x00                    # 正則: 厳密解・クリーン
    ypinv = LinearAlgebra.pinv(Lmat_alg(A16s, a_r.c)) * x_r.c  # 二証人: stdlib pinv
    @assert maximum(abs.(coeffs(yL) .- ypinv)) < 1e-8
    yR, _, _ = nsolve_right(A16s, a_r, x_r)
    dLR = maximum(abs.(coeffs(yL) .- coeffs(yR)))
    println("  正則a: a·y=x 残差 ", round(r1, sigdigits=2), " ✓ (pinv二証人一致) ; ",
            "左解≠右解: |L⁺x−R⁺x| = ", round(dLR, sigdigits=2), " (非可換の実測)")
    @assert dLR > 1e-3
    # 零因子: L 特異でも 解ける x は 厳密に・解けない x は SING で 正直に
    zd16 = Nel([i ∈ (4, 11) ? 1.0 : 0.0 for i in 1:16], 0x00)
    y0 = Nel(rand_vec(gs, 16), 0x00)
    x_in = tmul(A16s, zd16, y0)                                # range 内の x
    ys1, s1, _ = nsolve_left(A16s, zd16, x_in)
    @assert s1 < 1e-8 && flagof(ys1) == 0x00
    x_out = Nel(rand_vec(gs, 16), 0x00)                        # range 外の x (一般)
    ys2, t1, t2 = nsolve_left(A16s, zd16, x_out)
    @assert t1 > 1e-3 && t2 < 1e-6 && (flagof(ys2) & SING) != 0
    println("  零因子 e3+e10: range内→厳密解 ", round(s1, sigdigits=2), " ✓ ; ",
            "range外→SING+最小二乗(正規方程式 ", round(t2, sigdigits=2), ") ✓ 解けたフリなし")
    # 代数式 vs 方程式の解: conj-div y=(ā·x)/|a|² は 八元数まで解・セデニオンで破れる(実測)
    println("  conj-div (ā·x/|a|²) は ay=x の解か: ")
    for M in (4, 8, 16)
        Am = cd_alg(M); gm = _lcg()
        am = Nel(rand_vec(gm, M), 0x00); xm = Nel(rand_vec(gm, M), 0x00)
        cj = Nel(vcat(am.c[1], -am.c[2:end]), 0x00)
        ycj = tscale(tmul(Am, cj, xm), 1.0 / sum(abs2, am.c))
        rc = maximum(abs.(coeffs(tmul(Am, am, ycj)) .- xm.c))
        _, rs, _ = nsolve_left(Am, am, xm)
        println("    cd", M, ": conj-div残差 ", round(rc, sigdigits=2),
                (rc < 1e-8 ? " (解になる)" : " ✗(解でない)"),
                " / nsolve残差 ", round(rs, sigdigits=2))
        @assert (M <= 8) == (rc < 1e-8)                        # Artin: 八元数までは 2 元生成が結合的
        @assert rs < 1e-8
    end
    # 棚の任意の代数でも 同じ 1 本: dualquat で solve
    DQs = alg(:dualquat); gq2 = _lcg()
    aq = Nel(rand_vec(gq2, 8), 0x00); xq = Nel(rand_vec(gq2, 8), 0x00)
    _, rq2, _ = nsolve_left(DQs, aq, xq)
    @assert rq2 < 1e-8
    println("  棚の他代数 (dualquat): nsolve_left 残差 ", round(rq2, sigdigits=2), " ✓")
    # exp の家族 5 人: 左結合/右結合/対称化(jordan)/左作用/右作用 — 恒等式で結線
    ge_ = _lcg()
    A16e = cd_alg(16)
    ae = Nel(0.3 .* rand_vec(ge_, 16), 0x00)
    one16 = nel(A16e)
    dL = maximum(abs.(coeffs(nleft_action(A16e, ae, one16, 1.0)) .- coeffs(nexp(A16e, ae))))
    dR = maximum(abs.(coeffs(nright_action(A16e, ae, one16, 1.0)) .-
                      coeffs(nexp(A16e, ae; bracket = :right))))
    @assert dL < 1e-10 && dR < 1e-10           # 単位元に当てると 作用 = 結合 exp
    x0e = Nel(0.3 .* rand_vec(ge_, 16), 0x00)
    dt = 1e-6
    num = (coeffs(nleft_action(A16e, ae, x0e, dt)) .- coeffs(x0e)) ./ dt
    @assert maximum(abs.(num .- coeffs(tmul(A16e, ae, x0e)))) < 1e-3   # ẋ=a·x の有限差分検証
    js = nexp(jordan(A16e), ae)                # 対称化 exp = jordan 結合子経由(1 行)
    # 単一元では a∘a = a² ⟹ 対称化 exp ≡ 左結合 exp (べき結合律の潰れが jordan にも及ぶ。
    # 当初の「別物」assert は これに 反証された — 家族が 割れるのは 多元/行列でだけ、
    # は brackets 行(mat2⟨cd16⟩ 左vs右 0.012)で 既に 実測済み)
    @assert maximum(abs.(coeffs(js) .- coeffs(nexp(A16e, ae)))) < 1e-10
    println("  exp の家族5人: 作用(単位元)≡結合 ✓ ; ẋ=a·x 有限差分 ✓ ; ",
            "対称化も単一元では一致(潰れの法則) ✓ — L_a は exp(流れ)と L⁺(除算)の共有ハブ")
    # 除算の家族 5 人 (レビューの分類の完備) — conj-div の INEXACT は Hurwitz が押す
    gd = _lcg()
    for (M, solves) in ((4, true), (8, true), (16, false))
        Ad = cd_alg(M)
        ad_ = Nel(rand_vec(gd, M), 0x00); xd_ = Nel(rand_vec(gd, M), 0x00)
        yd, rd = nconj_div_left(Ad, ad_, xd_)
        @assert ((flagof(yd) & INEXACT) == 0) == solves "conj-div flag wrong at cd$M"
    end
    a4 = Nel(rand_vec(gd, 4), 0x00); x4 = Nel(rand_vec(gd, 4), 0x00)
    a16 = Nel(rand_vec(gd, 16), 0x00); x16 = Nel(rand_vec(gd, 16), 0x00)
    # ノルムの乗法性: 四元数では ‖conj_div‖=‖x‖/‖a‖・セデニオンでは 一般に 破れる(零因子の帰結)
    q4 = abs(sqrt(sum(abs2, coeffs(nconj_div_left(cd_alg(4), a4, x4)[1]))) - nnorm_div(x4, a4))
    q16 = abs(sqrt(sum(abs2, coeffs(nconj_div_left(cd_alg(16), a16, x16)[1]))) - nnorm_div(x16, a16))
    @assert q4 < 1e-12 && q16 > 1e-3
    nz = nnormalize(cd_alg(16), a16)
    @assert abs(sqrt(sum(abs2, coeffs(nz))) - 1.0) < 1e-12
    @assert all(coeffs(nnormalize(cd_alg(16), Nel(zeros(16), 0x00))) .== 0.0)   # 0/0 = 0
    println("  除算の家族5人: conj-div flag=Hurwitz発火(cd4/8 clean・cd16 INEXACT) ✓ ; ",
            "ノルム乗法性 ℍ ", round(q4, sigdigits=1), " / sed ", round(q16, sigdigits=2),
            " (零因子の帰結を実測) ✓ ; normalize ‖â‖=1・0→0 ✓")
    # Hurwitz の定理の実測: conj-div (ā/|a|²) が 逆元でいられる 機構は L_aᵀL_a = |a|²I
    # (合成代数 = ノルム乗法性)。共役=転置は 全 CD 次元で 成立するが、合成性は
    # ℝ,ℂ,ℍ,𝕆 (dim 1,2,4,8) で 尽きる — これが conj-div 門番の 正体。
    for (M, comp) in ((2, true), (4, true), (8, true), (16, false))
        Ah = cd_alg(M); gh = _lcg()
        avh = rand_vec(gh, M)
        Lh = Lmat_alg(Ah, avh)
        Lch = Lmat_alg(Ah, vcat(avh[1], -avh[2:end]))
        @assert maximum(abs.(Lh' .- Lch)) < 1e-12          # 共役 = 転置 (全次元)
        d2h = maximum(abs.(Lh' * Lh .- sum(abs2, avh) .* Matrix(LinearAlgebra.I, M, M)))
        @assert (d2h < 1e-9) == comp "Hurwitz boundary violated at cd$M"
    end
    println("  Hurwitz 実測: 共役=転置は全次元 ✓ ; L_aᵀL_a=|a|²I は dim 1,2,4,8 のみ",
            " (cd16 で破れ = conj-div 門番の機構) ✓")
    # 古典の証人団 (M₂(ℂ) = mat2⟨cd2⟩): 棚の演算を 古典理論と 突き合わせる
    MC = mat_over(cd_alg(2), 2)
    _at(r, c, k) = ((c - 1) * 2 + (r - 1)) * 2 + k
    _tocplx(v) = [complex(v[_at(r, c, 1)], v[_at(r, c, 2)]) for r in 1:2, c in 1:2]
    gc2 = _lcg()
    Ael = Nel(0.4 .* rand_vec(gc2, 8), 0x00)
    Em = _tocplx(coeffs(nexp(MC, Ael))); Am = _tocplx(coeffs(Ael))
    @assert abs(LinearAlgebra.det(Em) - exp(Am[1,1] + Am[2,2])) < 1e-10   # det∘exp = exp∘tr
    u2 = [complex(1.0, 0.5), complex(-0.3, 0.8)]; v2 = [complex(0.6, -0.2), complex(0.9, 0.4)]
    Acm = u2 * v2'                                          # rank-1 = 行列世界の日常的な零因子
    av = zeros(8)
    for r in 1:2, c in 1:2
        av[_at(r, c, 1)] = real(Acm[r, c]); av[_at(r, c, 2)] = imag(Acm[r, c])
    end
    xel = Nel(0.5 .* rand_vec(gc2, 8), 0x00)
    yel, _, rn2 = nsolve_left(MC, Nel(av, 0x00), xel)
    @assert rn2 < 1e-8 && (flagof(yel) & SING) != 0
    dcross = maximum(abs.(_tocplx(coeffs(yel)) .- LinearAlgebra.pinv(Acm) * _tocplx(coeffs(xel))))
    @assert dcross < 1e-6                                   # 実正則表現のpinv ≡ 複素の共役転置pinv
    println("  古典の証人団 (M₂(ℂ)): det∘exp=exp∘tr ✓ ; 特異solve ≡ 複素pinv (",
            round(dcross, sigdigits=2), ") ✓")
    # tape user-extensibility: a custom tape (Bessel-ish) runs on any Alg unchanged
    j0 = series(cd_alg(4), Nel(0.3 .* rand_vec(g, 4), 0x00),
                k -> iseven(k) ? Float64((-1)^(k ÷ 2) / (factorial(big(k ÷ 2))^2 * big(2)^k)) : 0.0)
    @assert all(isfinite, coeffs(j0))
    println("custom tape (user-defined coefficients) on cd4 ✓ — O layer is open, not an enum")
    println("done: cells × combinators × tapes compose freely; laws measured per combination")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    NestedSeries.self_test()
end
