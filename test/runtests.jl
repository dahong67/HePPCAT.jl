using HeteroscedasticPCA
using Test

module Reference
include("ref/alg,sage.jl")
include("ref/alg,mm.jl")
include("ref/alg,pgd.jl")
include("ref/alg,sgd.jl")
end

using Random, LinearAlgebra, BlockArrays
using Logging

# note: numerical discrepancies for root-finding v update seems to decrease
# to within default tolerance of ≈ by increasing the problem size.
# maybe some sort of conditioning issue. another solution (to add later) is
# to test single update steps instead of the whole sequence of iterates;
# issue seems to be an accumulation of drift.
rng = MersenneTwister(123)
@testset "alg,sage.jl" begin
    # nfull, vfull = (10, 40), (1, 4)
    # for d = 5:10:25, k = 1:3, L = 1:2
    nfull, vfull = (40, 10), (4, 1)
    for d = 50:25:100, k = 1:3, L = 1:2
        @testset "d=$d, k=$k, L=$L" begin
            n, v = nfull[1:L], vfull[1:L]
            F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
            Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

            Yflat = hcat(Y...)
            Yblock = BlockArray(Yflat, [d], collect(n))

            Yflatlist = collect(eachcol(Yflat))

            F0 = randn(rng, d, k)

            @testset "block" begin
                Uhat, λhat, vhat = HeteroscedasticPCA.ppca(Y, k, 10, F0, Val(:sage))
                vhat = [vcat(fill.(_vt,n)...) for _vt in vhat]
                Fref, vref = Reference.SAGE.ppca(Yblock, k, 10, F0)
                Frefsvd = svd.(Fref)
                Uref = getfield.(Frefsvd,:U)
                λref = [F.S.^2 for F in Frefsvd]
                @test Uhat ≈ Uref
                @test λhat ≈ λref
                @test vhat[2:end] ≈ vref[1:end-1]
            end

            @testset "flat" begin
                Uhat, λhat, vhat = HeteroscedasticPCA.ppca(Yflatlist, k, 10, F0, Val(:sage))
                Fref, vref = Reference.SAGE.ppca(Yflat, k, 10, F0)
                Frefsvd = svd.(Fref)
                Uref = getfield.(Frefsvd,:U)
                λref = [F.S.^2 for F in Frefsvd]
                @test Uhat ≈ Uref
                @test λhat ≈ λref
                @test vhat[2:end] ≈ vref[1:end-1]
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
            Yflatlist = collect(eachcol(Yflat))

            F0 = randn(rng, d, k)

            @testset "flat" begin
                Uhat, θ2hat, vhat = HeteroscedasticPCA.ppca(Yflatlist, k, 10, F0, Val(:mm))
                Fhat = Uhat[end] * Diagonal(sqrt.(θ2hat[end]))
                Fref, vref = Reference.MM.ppca(Yflat, k, 10, F0)
                @test Fhat ≈ Fref
                @test vhat[2:end] ≈ vref[1:end-1]
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
            Yflatlist = collect(eachcol(Yflat))

            F0 = randn(rng, d, k)

            @testset "flat" begin
                Uhat, θ2hat, vhat = HeteroscedasticPCA.ppca(Yflatlist, k, 10, F0, Val(:pgd))
                Fhat = Uhat[end] * Diagonal(sqrt.(θ2hat[end]))
                Fref, vref = Reference.PGD.ppca(Yflat, k, 10, F0)
                @test Fhat ≈ Fref
                @test vhat[2:end] ≈ vref[1:end-1]
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
            Yflatlist = collect(eachcol(Yflat))

            F0 = randn(rng, d, k)

            @testset "flat" begin
                Uhat, θ2hat, vhat = with_logger(NullLogger()) do
                    HeteroscedasticPCA.ppca(Yflatlist, k, 10, F0, Val(:sgd))
                end
                Fhat = Uhat[end] * Diagonal(sqrt.(θ2hat[end]))
                Fref, vref = Reference.SGD.ppca(Yflat, k, 10, F0)
                @test Fhat ≈ Fref
                @test vhat[2:end] ≈ vref[1:end-1]
            end
        end
    end
end
