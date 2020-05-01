using HeteroscedasticPCA
using Test

module Reference
include("ref/alg,sage.jl")
include("ref/alg,mm.jl")
include("ref/alg,pgd.jl")
include("ref/alg,sgd.jl")
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

# note: problem sizes that worked for SAGE seem to fail here
# would be interesting to investigate...
rng = MersenneTwister(123)
@testset "alg,mm.jl" begin
    nfull, vfull = (40, 10), (4, 1)
    for d = 50:25:100, k = 1:3, L = 1:2
        @testset "d=$d, k=$k, L=$L" begin
            n, v = nfull[1:L], vfull[1:L]
            F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
            Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

            Yflat = hcat(Y...)

            F0 = randn(rng, d, k)

            @testset "flat" begin
                Fhat, vhat = HeteroscedasticPCA.ppca(Yflat, k, 10, F0, Val(:mm))
                Fref, vref = Reference.MM.ppca(Yflat, k, 10, F0)
                @test Fhat == Fref
                @test vhat == vref
            end
        end
    end
end

# note: problem sizes that worked for SAGE seem to fail here
# would be interesting to investigate...
rng = MersenneTwister(123)
@testset "alg,pgd.jl" begin
    nfull, vfull = (40, 10), (4, 1)
    for d = 50:25:100, k = 1:3, L = 1:2
        @testset "d=$d, k=$k, L=$L" begin
            n, v = nfull[1:L], vfull[1:L]
            F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
            Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

            Yflat = hcat(Y...)

            F0 = randn(rng, d, k)

            @testset "flat" begin
                Fhat, vhat = HeteroscedasticPCA.ppca(Yflat, k, 10, F0, Val(:pgd))
                Fref, vref = Reference.PGD.ppca(Yflat, k, 10, F0)
                @test Fhat == Fref
                @test vhat == vref
            end
        end
    end
end

# note: problem sizes that worked for SAGE seem to fail here
# would be interesting to investigate...seemed to do better
# than mm or pgd though
rng = MersenneTwister(123)
@testset "alg,sgd.jl" begin
    nfull, vfull = (40, 10), (4, 1)
    for d = 50:25:100, k = 1:3, L = 1:2
        @testset "d=$d, k=$k, L=$L" begin
            n, v = nfull[1:L], vfull[1:L]
            F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
            Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

            Yflat = hcat(Y...)

            F0 = randn(rng, d, k)

            @testset "flat" begin
                Fhat, vhat = HeteroscedasticPCA.ppca(Yflat, k, 10, F0, Val(:sgd))
                Fref, vref = Reference.SGD.ppca(Yflat, k, 10, F0)
                @test Fhat == Fref
                @test vhat == vref
            end
        end
    end
end
