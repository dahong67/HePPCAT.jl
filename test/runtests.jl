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

rng = MersenneTwister(123)
@testset "alg,sage.jl" begin
    nfull, vfull = (10, 40), (1, 4)
    @testset "d=$d, k=$k, L=$L" for d = 5:10:25, k = 1:3, L = 1:2
        n, v = nfull[1:L], vfull[1:L]
        F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
        Yblock = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

        Yflatlist = collect(eachcol(hcat(Yblock...)))

        F0 = randn(rng, d, k)
        iters = 4

        @testset "block" begin
            Fref = Vector{typeof(F0)}(undef,iters+1)
            vref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            Fref[1] = copy(F0)
            for t in 1:iters
                vref[t] = Reference.SAGE.updatev(Fref[t],Yblock)
                Fref[t+1] = Reference.SAGE.updateF(Fref[t],vref[t],Yblock)
            end
            vref[end] = Reference.SAGE.updatev(Fref[end],Yblock)
            
            Fsvd = svd.(Fref)
            @testset "updateF! (ExpectationMaximization)" begin
                for t in 2:length(Fref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Fsvd[t-1].U),
                        copy(Fsvd[t-1].S.^2),
                        copy(Fsvd[t-1].Vt),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updateF!(M,Yblock,HeteroscedasticPCA.ExpectationMaximization())
                    @test M.U ≈ Fsvd[t].U
                    @test M.λ ≈ Fsvd[t].S.^2
                    @test M.Vt ≈ Fsvd[t].Vt
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Fsvd[t].U),
                        copy(Fsvd[t].S.^2),
                        copy(Fsvd[t].Vt),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updatev!(M,Yblock,HeteroscedasticPCA.RootFinding())
                    @test M.v ≈ vref[t]
                end
            end
        end

        # updateF! seems to disagree at latter iterations
        @testset "flat" begin
            Fref = Vector{typeof(F0)}(undef,iters+1)
            vref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            Fref[1] = copy(F0)
            for t in 1:iters
                vref[t] = Reference.SAGE.updatev(Fref[t],Yflatlist)
                Fref[t+1] = Reference.SAGE.updateF(Fref[t],vref[t],Yflatlist)
            end
            vref[end] = Reference.SAGE.updatev(Fref[end],Yflatlist)
            
            Fsvd = svd.(Fref)
            @testset "updateF! (ExpectationMaximization)" begin
                for t in 2:length(Fref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Fsvd[t-1].U),
                        copy(Fsvd[t-1].S.^2),
                        copy(Fsvd[t-1].Vt),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updateF!(M,Yflatlist,HeteroscedasticPCA.ExpectationMaximization())
                    @test M.U ≈ Fsvd[t].U
                    @test M.λ ≈ Fsvd[t].S.^2
                    @test M.Vt ≈ Fsvd[t].Vt
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Fsvd[t].U),
                        copy(Fsvd[t].S.^2),
                        copy(Fsvd[t].Vt),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updatev!(M,Yflatlist,HeteroscedasticPCA.RootFinding())
                    @test M.v ≈ vref[t]
                end
            end
        end
    end
end

# note: problem sizes that worked for SAGE seem to fail here
# would be interesting to investigate...
rng = MersenneTwister(123)
@testset "alg,mm.jl" begin
    nfull, vfull = (40, 10), (4, 1)
    @testset "d=$d, k=$k, L=$L" for d = 50:25:100, k = 1:3, L = 1:2
        n, v = nfull[1:L], vfull[1:L]
        F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
        Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

        Yflat = hcat(Y...)
        Yflatlist = collect(eachcol(Yflat))

        F0 = randn(rng, d, k)
        iters = 4

        @testset "flat" begin
            Uref = Vector{typeof(F0)}(undef,iters+1)
            λref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            vref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            Q,S,_ = svd(F0)
            Uref[1] = Q[:,1:k]
            λref[1] = S[1:k].^2
            for t in 1:iters
                vref[t] = Reference.MM.updatev(Uref[t],λref[t],Yflat)
                λref[t+1] = Reference.MM.updateθ2(Uref[t],vref[t],Yflat)
                Uref[t+1] = Reference.MM.updateU(Uref[t],λref[t+1],vref[t],Yflat)
            end
            vref[end] = Reference.MM.updatev(Uref[end],λref[end],Yflat)
            
            VtI = Matrix{Float64}(I,k,k)
            @testset "updateλ! (RootFinding)" begin
                for t in 2:length(λref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t-1]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updateλ!(M,Yflatlist,HeteroscedasticPCA.RootFinding())
                    @test M.λ ≈ λref[t]
                end
            end
            @testset "updateU! (MinorizeMaximize)" begin
                for t in 2:length(Uref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updateU!(M,Yflatlist,HeteroscedasticPCA.MinorizeMaximize())
                    @test M.U ≈ Uref[t]
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Uref[t]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updatev!(M,Yflatlist,HeteroscedasticPCA.RootFinding())
                    @test M.v ≈ vref[t]
                end
            end
        end
    end
end

# note: problem sizes that worked for SAGE seem to fail here
# would be interesting to investigate...
rng = MersenneTwister(123)
@testset "alg,pgd.jl" begin
    nfull, vfull = (40, 10), (4, 1)
    @testset "d=$d, k=$k, L=$L" for d = 50:25:100, k = 1:3, L = 1:2
        n, v = nfull[1:L], vfull[1:L]
        F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
        Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

        Yflat = hcat(Y...)
        Yflatlist = collect(eachcol(Yflat))

        F0 = randn(rng, d, k)
        iters = 4

        @testset "flat" begin
            Uref = Vector{typeof(F0)}(undef,iters+1)
            λref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            vref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            Q,S,_ = svd(F0)
            Uref[1] = Q[:,1:k]
            λref[1] = S[1:k].^2
            Ynorms = Reference.PGD.computeYcolnorms(Yflat)
            for t in 1:iters
                vref[t] = Reference.PGD.updatev(Uref[t],λref[t],Yflat)
                λref[t+1] = Reference.PGD.updateθ2(Uref[t],vref[t],Yflat)
                L = Reference.PGD.updateL(Ynorms,λref[t+1],vref[t])
                Uref[t+1] = Reference.PGD.updateU(Uref[t],λref[t+1],vref[t],Yflat,1/L)
            end
            vref[end] = Reference.PGD.updatev(Uref[end],λref[end],Yflat)
            
            VtI = Matrix{Float64}(I,k,k)
            @testset "updateλ! (RootFinding)" begin
                for t in 2:length(λref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t-1]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updateλ!(M,Yflatlist,HeteroscedasticPCA.RootFinding())
                    @test M.λ ≈ λref[t]
                end
            end
            @testset "updateU! (ProjectedGradientAscent)" begin
                for t in 2:length(Uref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    Ynorms = vec(mapslices(norm,Yflat,dims=1))
                    L = sum(ynorm^2*maximum([λj/vi/(λj+vi) for λj in M.λ]) for (ynorm,vi) in zip(Ynorms,M.v))
                    HeteroscedasticPCA.updateU!(M,Yflatlist,HeteroscedasticPCA.ProjectedGradientAscent(1/L))
                    @test M.U ≈ Uref[t]
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Uref[t]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updatev!(M,Yflatlist,HeteroscedasticPCA.RootFinding())
                    @test M.v ≈ vref[t]
                end
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
    @testset "d=$d, k=$k, L=$L" for d = 50:25:100, k = 1:3, L = 1:2
        n, v = nfull[1:L], vfull[1:L]
        F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
        Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

        Yflat = hcat(Y...)
        Yflatlist = collect(eachcol(Yflat))

        F0 = randn(rng, d, k)
        iters = 4
        max_line = 50
        α = 0.8
        β = 0.5
        σ = 1
        
        @testset "flat" begin
            Uref = Vector{typeof(F0)}(undef,iters+1)
            λref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            vref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            Q,S,_ = svd(F0)
            Uref[1] = Q[:,1:k]
            λref[1] = S[1:k].^2

            for t in 1:iters
                vref[t] = Reference.SGD.updatev(Uref[t],λref[t],Yflat)
                λref[t+1] = Reference.SGD.updateθ2(Uref[t],vref[t],Yflat)
                Uref[t+1] = Reference.SGD.updateU(Uref[t],λref[t+1],vref[t],Yflat,α,β,σ,max_line,t)
            end
            vref[end] = Reference.SGD.updatev(Uref[end],λref[end],Yflat)
            
            VtI = Matrix{Float64}(I,k,k)
            @testset "updateλ! (RootFinding)" begin
                for t in 2:length(λref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t-1]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updateλ!(M,Yflatlist,HeteroscedasticPCA.RootFinding())
                    @test M.λ ≈ λref[t]
                end
            end
            @testset "updateU! (StiefelGradientAscent)" begin
                for t in 2:length(Uref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updateU!(M,Yflatlist,HeteroscedasticPCA.StiefelGradientAscent(50,0.8,0.5,1.0))
                    @test M.U ≈ Uref[t]
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HeteroscedasticPCA.HPPCA(
                        copy(Uref[t]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    HeteroscedasticPCA.updatev!(M,Yflatlist,HeteroscedasticPCA.RootFinding())
                    @test M.v ≈ vref[t]
                end
            end
        end
    end
end

@testset "F/gradF" begin
    nfull, vfull = (40, 10), (4, 1)
    d, k = 25, 3
    for L = 1:2
        n, v = nfull[1:L], vfull[1:L]
        F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
        Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

        Yflat = collect(eachcol(hcat(Y...)))
        F0 = randn(rng, d, k)

        Uhat, λhat, vhat = HeteroscedasticPCA.ppca(Y, k, 10, F0, Val(:sage))
        @test all(zip(Uhat[2:end],λhat[2:end],vhat[2:end])) do (U,λ,vv)
            blockF = HeteroscedasticPCA.F(U, λ, vv, Y)
            flatF  = HeteroscedasticPCA.F(U, λ, vcat(fill.(vv,n)...), Yflat)
            blockF ≈ flatF
        end
        @test all(zip(Uhat[2:end],λhat[2:end],vhat[2:end])) do (U,λ,vv)
            blockg = HeteroscedasticPCA.gradF(U, λ, vv, Y)
            flatg  = HeteroscedasticPCA.gradF(U, λ, vcat(fill.(vv,n)...), Yflat)
            blockg ≈ flatg
        end
    end
end

@testset "updateλ! (RootFinding, block)" begin
    nfull, vfull = (40, 10), (4, 1)
    d, k = 25, 3
    for L = 1:2
        n, v = nfull[1:L], vfull[1:L]
        F, Z = randn(rng, d, k), [randn(rng, k, nl) for nl in n]
        Y = [F * Zl + sqrt(vl) * randn(rng, d, nl) for (Zl, vl, nl) in zip(Z, v, n)]

        Yflat = collect(eachcol(hcat(Y...)))
        F0 = randn(rng, d, k)

        Uhat, λhat, vhat = HeteroscedasticPCA.ppca(Y, k, 10, F0, Val(:sage))
        VtI = Matrix{Float64}(I,k,k)
        @test all(zip(Uhat[2:end],λhat[2:end],vhat[2:end])) do (U,λ,vv)
            Mblock = HeteroscedasticPCA.HPPCA(copy(U), copy(λ), copy(VtI), copy(vv))
            HeteroscedasticPCA.updateλ!(Mblock,Y,HeteroscedasticPCA.RootFinding())
            Mflat = HeteroscedasticPCA.HPPCA(copy(U), copy(λ), copy(VtI), vcat(fill.(copy(vv),n)...))
            HeteroscedasticPCA.updateλ!(Mflat,Yflat,HeteroscedasticPCA.RootFinding())
            Mblock.λ ≈ Mflat.λ
        end
    end
end
