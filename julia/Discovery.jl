# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
Discovery — 暗黙形の物理法則発見、全部この棚から (formal 版; demo_discovery.jl の 一般化)。

  法則 = Σc·(単項式/微分項) = 0 の 零空間。y=f(x) 明示形の「√(和)の壁」(相対論のγ等)は
  目的変数を ライブラリに 入れた 瞬間に 消える。解は SVD 一発(勾配法ゼロ・シード概念なし)。
  自明解は 構造で 塞ぐ: c=0 は 球面 ‖c‖=1(SVD)・恒真式 x^w−x^w は 離散格子・盆地は 大域解。

  微分は 外部ライブラリ不要 — **棚が微分ライブラリ**:
    HD = Λ1⊗Λ1 (超双対数)。`HDNum` が 普通の Julia 演算子を 提供し、
    f(s+n) = f(s) + f′(s)n + f″(s)n²/2  (n 冪零 ⟹ 厳密に 切れる = `_lift`)
    で 任意の 初等関数式から ψ′, ψ″ が 厳密値(誤差 0.0)で 出る。
    (ForwardDiff.jl の 中身も 双対数 — 同じ数学を 棚の 結合子で 組んだ形)

  毒(センサ飽和 Inf・欠測 NaN)は ScalarTot の 監査済み入口が 名指し → 行除外。
  超双対評価は 点ごと 独立なので 毒は 拡散しない(有限差分は 隣に 塗り広げる)。

  実測(self_test で 回帰化): 調和振動子 E=0.5 を 15桁・毒10点を 10行のまま 名指し・
  水素 E=−0.5・法則なしデータには σ_min 大で 正直に「なし」。Python 双子は
  varpro-powersum-nn/implicit_discovery.py (複素対応は そちらが 先行)。
"""
module Discovery

import LinearAlgebra
import Random

# 依存の解決(vendor しない・本家 total-arith-cuda から 取る):
#   ① ENV["TOTAL_ARITH_JULIA"] ② 自分の隣(total-arith-cuda 内での実行)
#   ③ 隣に clone された ../total-arith-cuda/julia (varpro-powersum-nn 等からの実行)
function _dep(f::String)
    for d in (get(ENV, "TOTAL_ARITH_JULIA", ""), @__DIR__,
              joinpath(@__DIR__, "..", "..", "total-arith-cuda", "julia"))
        isempty(d) && continue
        p = joinpath(d, f)
        isfile(p) && return p
    end
    error("依存 $f が見つかりません。total-arith-cuda を 隣に clone するか " *
          "ENV[\"TOTAL_ARITH_JULIA\"] に その julia/ ディレクトリを 指定してください。")
end
include(_dep("NestedSeries.jl")); using .NestedSeries
include(_dep("ScalarTot.jl"));   using .ScalarTot

export HDNum, hdvar, value, d1, d2, build_library, discover, fmt_law

# ================================================================ 微分 = 棚 (Λ1⊗Λ1)
const HD = tensor(grassmann_alg(1), grassmann_alg(1))    # [1, ε₂, ε₁, ε₁ε₂] 4次元

"超双対数: 棚の Nel を 包み、普通の演算子を 提供する"
struct HDNum
    z::Nel
end
hdvar(x::Real) = HDNum(nel(HD, [Float64(x), 1.0, 1.0, 0.0]))   # x + ε₁ + ε₂
HDNum(x::Real) = HDNum(nel(HD, [Float64(x), 0.0, 0.0, 0.0]))
value(a::HDNum) = coeffs(a.z)[1]
d1(a::HDNum)    = coeffs(a.z)[2]                          # f′ (ε₁ 成分)
d2(a::HDNum)    = coeffs(a.z)[4]                          # f″ (ε₁ε₂ 成分)

Base.:+(a::HDNum, b::HDNum) = HDNum(NestedSeries.tadd(a.z, b.z))
Base.:-(a::HDNum, b::HDNum) = HDNum(NestedSeries.tadd(a.z, NestedSeries.tscale(b.z, -1.0)))
Base.:-(a::HDNum) = HDNum(NestedSeries.tscale(a.z, -1.0))
Base.:*(a::HDNum, b::HDNum) = HDNum(NestedSeries.tmul(HD, a.z, b.z))
Base.:*(s::Real, a::HDNum) = HDNum(NestedSeries.tscale(a.z, Float64(s)))
Base.:*(a::HDNum, s::Real) = s * a
Base.:+(a::HDNum, s::Real) = a + HDNum(s)
Base.:+(s::Real, a::HDNum) = a + HDNum(s)
Base.:-(a::HDNum, s::Real) = a + HDNum(-s)
Base.:-(s::Real, a::HDNum) = HDNum(s) - a
Base.:/(a::HDNum, s::Real) = HDNum(NestedSeries.tscale(a.z, 1.0 / s))

"""万能鍵: f(s+n) = f(s) + f′(s)·n + f″(s)·n²/2 — n は 冪零(n³=0)なので **厳密**。
   初等関数は (f, f′, f″) の 3 つ組を 渡すだけで 超双対に 持ち上がる。"""
function _lift(a::HDNum, f, f1, f2)
    s = value(a)
    n = Nel([0.0; coeffs(a.z)[2:end]], NestedSeries.flagof(a.z))
    n2 = NestedSeries.tmul(HD, n, n)
    acc = nel(HD, [f(s), 0, 0, 0])
    acc = NestedSeries.tadd(acc, NestedSeries.tscale(n, f1(s)))
    HDNum(NestedSeries.tadd(acc, NestedSeries.tscale(n2, f2(s) / 2)))
end
Base.exp(a::HDNum)  = _lift(a, exp,  exp,  exp)
Base.log(a::HDNum)  = _lift(a, log,  s -> 1/s, s -> -1/s^2)
Base.sin(a::HDNum)  = _lift(a, sin,  cos,  s -> -sin(s))
Base.cos(a::HDNum)  = _lift(a, cos,  s -> -sin(s), s -> -cos(s))
Base.sqrt(a::HDNum) = _lift(a, sqrt, s -> 0.5/sqrt(s), s -> -0.25/s^1.5)
Base.inv(a::HDNum)  = _lift(a, s -> 1/s, s -> -1/s^2, s -> 2/s^3)
Base.:/(a::HDNum, b::HDNum) = a * inv(b)
Base.:/(s::Real, a::HDNum) = s * inv(a)
Base.:^(a::HDNum, p::Real) = _lift(a, s -> s^p, s -> p*s^(p-1), s -> p*(p-1)*s^(p-2))
Base.:^(a::HDNum, p::Integer) = a^Float64(p)   # 統一(冪零打ち切りで厳密)

"f: HDNum→HDNum の 関数から (f(x), f′(x), f″(x)) を 厳密値で"
derivs(f, x::Real) = (h = f(hdvar(x)); (value(h), d1(h), d2(h)))

# ================================================================ 発見: 零空間 + フラグ
function build_library(vars::Dict{String,<:AbstractVector}, max_deg::Int;
                       extra::Union{Dict{String,<:AbstractVector},Nothing} = nothing)
    names = sort(collect(keys(vars)))
    cols = Vector{Vector{Float64}}(); labels = String[]
    if !isempty(names)
        data = [Float64.(vars[n]) for n in names]
        nvar = length(names)
        for e in Iterators.product(fill(0:max_deg, nvar)...)
            sum(e) > max_deg && continue
            push!(cols, reduce(.*, [data[i] .^ e[i] for i in 1:nvar];
                               init = ones(length(data[1]))))
            lab = join([e[i] > 1 ? "$(names[i])^$(e[i])" : names[i]
                        for i in 1:nvar if e[i] > 0], "·")
            push!(labels, isempty(lab) ? "1" : lab)
        end
    end
    if extra !== nothing
        for k in sort(collect(keys(extra)))
            push!(cols, Float64.(extra[k])); push!(labels, k)
        end
    end
    reduce(hcat, cols), labels
end

"""暗黙法則の発見: (law, σ_min, gap, n_flagged, c, labels)。
   フラグ=ScalarTot 監査済み入口(毒の名指し)・値=元 Float64(精度無傷)・解=零空間 SVD。"""
function discover(vars::Dict{String,<:AbstractVector}; max_deg::Int = 2,
                  extra = nothing)
    lib, labels = build_library(vars, max_deg; extra)
    flagged = falses(size(lib, 1))
    for i in axes(lib, 1), j in axes(lib, 2)
        flagged[i] |= ScalarTot.isflagged(TotNum(lib[i, j]))
    end
    clean = lib[.!flagged, :]
    norms = [LinearAlgebra.norm(c) for c in eachcol(clean)]
    norms[norms .== 0] .= 1.0
    _, S, V = LinearAlgebra.svd(clean ./ norms')
    c = V[:, end] ./ norms
    c = c ./ c[argmax(abs.(c))]
    law = [(co, lab) for (co, lab) in zip(c, labels) if abs(co) > 1e-4]
    (law = law, sigma_min = S[end], gap = S[end-1] / max(S[end], 1e-300),
     n_flagged = count(flagged), c = c, labels = labels)
end

fmt_law(r) = "0 = " * join(["$(round(co, sigdigits=7))·$lab" for (co, lab) in r.law], "  +  ")

# ================================================================ self-test
function self_test()
    println("Discovery — 棚だけで 法則発見 (微分=Λ1⊗Λ1・毒=ScalarTot・解=零空間)")

    # ① HDNum の 厳密性: 合成関数でも 誤差ゼロ級
    f(x) = sin(x) * exp(x) + x^3 / (1 + x^2)
    x0 = 0.8
    v, g1, g2 = derivs(f, x0)
    h = 1e-5                                       # 検証用の 有限差分(真値の 代わり)
    fd1 = (f(hdvar(x0 + h)) |> value) - (f(hdvar(x0 - h)) |> value)
    @assert abs(v - (sin(x0)exp(x0) + x0^3/(1+x0^2))) < 1e-14
    @assert abs(g1 - fd1 / 2h) < 1e-6              # 有限差分とは 一致・厳密性は ↓の物理で
    println("  ① HDNum: sin·exp + x³/(1+x²) の 値/微分 ✓ (普通の構文で 厳密ジェット)")

    # ② 調和振動子 + 毒: E=0.5 を 高精度で・毒は 拡散せず 名指し
    xs = collect(range(-4, 4, length = 4000))
    U = zeros(4000); U2 = zeros(4000)
    for (i, x) in enumerate(xs)
        p = exp(-(hdvar(x)^2) / 2)
        U[i] = value(p); U2[i] = d2(p)
    end
    pois = Random.shuffle(Random.MersenneTwister(0), 1:4000)[1:10]
    U[pois[1:5]] .= Inf; U[pois[6:10]] .= NaN
    r = discover(Dict{String,Vector{Float64}}();
                 extra = Dict("ψ''" => U2, "ψ" => U, "x·ψ" => xs .* U, "x²ψ" => xs.^2 .* U))
    d = Dict(lab => co for (co, lab) in r.law)
    E = d["ψ"] / d["ψ''"] / 2
    @assert r.n_flagged == 10 && abs(E - 0.5) < 1e-10 && r.sigma_min < 1e-12
    println("  ② 調和振動子+毒10点: ", r.n_flagged, "行のみ除外(拡散なし)・E = ", E,
            " ・σ_min ", round(r.sigma_min, sigdigits = 2), " ✓")

    # ③ 水素原子: u = r·e^{-r} → u'' + 2u/r − u = 0, E = −1/2
    rs = collect(range(0.05, 12, length = 4000))
    Uh = zeros(4000); Uh2 = zeros(4000)
    for (i, rr) in enumerate(rs)
        hu = hdvar(rr) * exp(-hdvar(rr))
        Uh[i] = value(hu); Uh2[i] = d2(hu)
    end
    r = discover(Dict{String,Vector{Float64}}();
                 extra = Dict("u''" => Uh2, "u/r" => Uh ./ rs, "u" => Uh, "r·u" => rs .* Uh))
    d = Dict(lab => co for (co, lab) in r.law)
    Eh = d["u"] / d["u''"] / 2
    @assert abs(Eh + 0.5) < 1e-10
    println("  ③ 水素原子: E = ", Eh, " ✓ (理論 −1/2)")

    # ④ 法則なしデータ: 正直に「なし」
    rng = Random.MersenneTwister(1)
    r = discover(Dict("a" => rand(rng, 2000) .+ 0.1, "b" => rand(rng, 2000) .+ 0.1,
                      "c" => rand(rng, 2000) .+ 0.1); max_deg = 2)
    @assert r.sigma_min > 1e-3
    println("  ④ 無関係データ: σ_min ", round(r.sigma_min, sigdigits = 2),
            " = 法則なしを 正直に 報告 ✓")
    println("done — 微分も 全域化も 発見も、この 棚 ひとつから")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Discovery.self_test()
end
