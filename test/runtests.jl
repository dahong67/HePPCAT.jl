using HeteroscedasticPCA
using Test

module Reference
include("ref/alg,sage.jl")
end

using Random, LinearAlgebra, BlockArrays

rng = MersenneTwister(123)

@testset "alg,sage.jl" begin
    nfull, vfull = (10, 40), (1, 4)
    for d = 5:10:25, k = 1:3, L = 1:2
        @testset "d=$d, k=$k, L=$L" begin
            n, v = nfull[1:L], vfull[1:L]
            F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
            Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

            Yflat = hcat(Y...)
            Yblock = BlockArray(Yflat, [d], collect(n))

            Yflatlist = collect(eachcol(Yflat))

            F0 = randn(rng, d, k)

            @testset "block" begin
                Fhat, vhat = HeteroscedasticPCA.ppca(Y, k, 10, F0, Val(:sage))
                Fref, vref = Reference.SAGE.ppca(Yblock, k, 10, F0)
                @test Fhat == Fref
                @test [vcat(fill.(_vt,n)...) for _vt in vhat] == vref
            end

            @testset "flat" begin
                Fhat, vhat = HeteroscedasticPCA.ppca(Yflatlist, k, 10, F0, Val(:sage))
                Fref, vref = Reference.SAGE.ppca(Yflat, k, 10, F0)
                @test Fhat == Fref
                @test vhat == vref
            end
        end
    end
end
