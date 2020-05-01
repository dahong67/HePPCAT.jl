using BenchmarkTools, HeteroscedasticPCA
using Random, LinearAlgebra

const SUITE = BenchmarkGroup()

rng = MersenneTwister(123)

# SAGE
SUITE["sage"] = BenchmarkGroup()
nfull, vfull = (10, 40), (1, 4)
for d = 5:10:15, k = 1:3:4, L = 1:2
    n, v = nfull[1:L], vfull[1:L]
    F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
    Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

    Yflat = collect.(eachcol(hcat(Y...)))
    F0 = randn(rng, d, k)

    for T = 0:10:20
        SUITE["sage"]["block: d=$d, k=$k, L=$L, T=$T"] = @benchmarkable HeteroscedasticPCA.ppca($Y, $k, $T, $F0, $(Val(:sage)))
        SUITE["sage"]["flat:  d=$d, k=$k, L=$L, T=$T"] = @benchmarkable HeteroscedasticPCA.ppca($Yflat, $k, $T, $F0, $(Val(:sage)))
    end
end
