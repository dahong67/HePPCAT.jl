using HeteroscedasticPCA
using LinearAlgebra, Random, Test

# Relevant functions
using HeteroscedasticPCA: HPPCA
using HeteroscedasticPCA: ExpectationMaximization, MinorizeMaximize,
    ProjectedGradientAscent, RootFinding, StiefelGradientAscent
using HeteroscedasticPCA: updateF!, updatev!, updateU!, updateλ!

# Load reference implementations
include("ref.jl")

# Convenience functions
factormatrix(M::HPPCA) = M.U*sqrt(Diagonal(M.λ))*M.Vt
flatten(M::HPPCA,n) = HPPCA(M.U,M.λ,M.Vt,vcat(fill.(M.v,n)...))

T = 5
n, v = (40, 10), (4, 1)
@testset "n=$(n[1:L]), v=$(v[1:L]), d=$d, k=$k" for L = 1:2, d = 5:10:25, k = 1:3
    # Compute test problem
    rng = MersenneTwister(123)
    F, Z = randn(rng,d,k), [randn(rng,k,n[l]) for l in 1:L]
    Yb = [F*Z[l] + sqrt(v[l])*randn(rng,d,n[l]) for l in 1:L]   # blocked
    Yf = collect(eachcol(hcat(Yb...)))                          # flatten
    
    @testset "block calls" begin
        # Generate sequence of test iterates
        init = svd(randn(rng,d,k))
        MM = [HPPCA(init.U,init.S.^2,init.Vt,rand(rng,L))]
        for t in 1:T
            push!(MM, deepcopy(MM[end]))
            updatev!(MM[end],Yb,RootFinding())
            updateF!(MM[end],Yb,ExpectationMaximization())
        end
        
        # Test v updates
        @testset "updatev! (RootFinding): t=$t" for t in 1:T
            vr = Ref.updatev_roots(MM[t].U,MM[t].λ,Yb)
            Mb = updatev!(deepcopy(MM[t]),Yb,RootFinding())
            @test vr ≈ Mb.v
        end
        
        # Test F updates
        @testset "updateF! (ExpectationMaximization): t=$t" for t in 1:T
            Fr = Ref.updateF_em(factormatrix(MM[t]),MM[t].v,Yb)
            Mb = updateF!(deepcopy(MM[t]),Yb,ExpectationMaximization())
            @test Fr ≈ factormatrix(Mb)
            Mf = updateF!(flatten(deepcopy(MM[t]),n[1:L]),Yf,ExpectationMaximization())
            @test Fr ≈ factormatrix(Mf)
        end
        
        # Test U updates
        @testset "updateU! (MinorizeMaximize): t=$t" for t in 1:T
            Ur = Ref.updateU_mm(MM[t].U,MM[t].λ,MM[t].v,Yb)
            Mb = updateU!(deepcopy(MM[t]),Yb,MinorizeMaximize())
            @test Ur ≈ Mb.U
            Mf = updateU!(flatten(deepcopy(MM[t]),n[1:L]),Yf,MinorizeMaximize())
            @test Ur ≈ Mf.U
        end
        @testset "updateU! (ProjectedGradientAscent): t=$t" for t in 1:T
            Lip = sum(norm(Yl)^2*maximum(λj/vl/(λj+vl) for λj in MM[t].λ) for (Yl,vl) in zip(Yb,MM[t].v))
            Ur = Ref.updateU_pga(MM[t].U,MM[t].λ,MM[t].v,Yb,1/Lip)
            Mb = updateU!(deepcopy(MM[t]),Yb,ProjectedGradientAscent(1/Lip))
            @test Ur ≈ Mb.U
            Mf = updateU!(flatten(deepcopy(MM[t]),n[1:L]),Yf,ProjectedGradientAscent(1/Lip))
            @test Ur ≈ Mf.U
        end
        @testset "updateU! (StiefelGradientAscent): t=$t" for t in 1:T
            Ur = Ref.updateU_sga(MM[t].U,MM[t].λ,MM[t].v,Yb,50,0.8,0.5,1.0)
            Mb = updateU!(deepcopy(MM[t]),Yb,StiefelGradientAscent(50,0.8,0.5,1.0))
            @test Ur ≈ Mb.U
            Mf = updateU!(flatten(deepcopy(MM[t]),n[1:L]),Yf,StiefelGradientAscent(50,0.8,0.5,1.0))
            @test Ur ≈ Mf.U
        end
        
        # Test λ updates
        @testset "updateλ! (RootFinding): t=$t" for t in 1:T
            λr = Ref.updateλ_roots(MM[t].U,MM[t].v,Yb)
            Mb = updateλ!(deepcopy(MM[t]),Yb,RootFinding())
            @test λr ≈ Mb.λ
            Mf = updateλ!(flatten(deepcopy(MM[t]),n[1:L]),Yf,RootFinding())
            @test λr ≈ Mf.λ
        end
        
        # Test F/gradF
        @testset "F/gradF: t=$t" for t in 1:T
            vf = vcat(fill.(MM[t].v,n[1:L])...)
            Fr = Ref.F(MM[t].U,MM[t].λ,vf,Yf)
            Fb = HeteroscedasticPCA.F(MM[t].U,MM[t].λ,MM[t].v,Yb)
            @test Fr ≈ Fb
            Ff = HeteroscedasticPCA.F(MM[t].U,MM[t].λ,vf,Yf)
            @test Fr ≈ Ff
            
            Gr = Ref.gradF(MM[t].U,MM[t].λ,vf,Yf)
            Gb = HeteroscedasticPCA.gradF(MM[t].U,MM[t].λ,MM[t].v,Yb)
            @test Gr ≈ Gb
            Gf = HeteroscedasticPCA.gradF(MM[t].U,MM[t].λ,vf,Yf)
            @test Gr ≈ Gf
        end
    end
    
    @testset "flat calls" begin
        # Generate sequence of test iterates
        init = svd(randn(rng,d,k))
        MM = [HPPCA(init.U,init.S.^2,init.Vt,rand(rng,sum(n[1:L])))]
        for t in 1:T
            push!(MM, deepcopy(MM[end]))
            updatev!(MM[end],Yf,RootFinding())
            updateF!(MM[end],Yf,ExpectationMaximization())
        end
        
        # Test v updates
        @testset "updatev! (RootFinding): t=$t" for t in 1:T
            vr = Ref.updatev_roots(MM[t].U,MM[t].λ,Yf)
            Mf = updatev!(deepcopy(MM[t]),Yf,RootFinding())
            @test vr ≈ Mf.v
        end
        
        # Test F updates
        @testset "updateF! (ExpectationMaximization): t=$t" for t in 1:T
            Fr = Ref.updateF_em(factormatrix(MM[t]),MM[t].v,Yf)
            Mf = updateF!(deepcopy(MM[t]),Yf,ExpectationMaximization())
            @test Fr ≈ factormatrix(Mf)
        end
        
        # Test U updates
        @testset "updateU! (MinorizeMaximize): t=$t" for t in 1:T
            Ur = Ref.updateU_mm(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Mf = updateU!(deepcopy(MM[t]),Yf,MinorizeMaximize())
            @test Ur ≈ Mf.U
        end
        @testset "updateU! (ProjectedGradientAscent): t=$t" for t in 1:T
            Lip = sum(norm(Yl)^2*maximum(λj/vl/(λj+vl) for λj in MM[t].λ) for (Yl,vl) in zip(Yf,MM[t].v))
            Ur = Ref.updateU_pga(MM[t].U,MM[t].λ,MM[t].v,Yf,1/Lip)
            Mf = updateU!(deepcopy(MM[t]),Yf,ProjectedGradientAscent(1/Lip))
            @test Ur ≈ Mf.U
        end
        @testset "updateU! (StiefelGradientAscent): t=$t" for t in 1:T
            Ur = Ref.updateU_sga(MM[t].U,MM[t].λ,MM[t].v,Yf,50,0.8,0.5,1.0)
            Mf = updateU!(deepcopy(MM[t]),Yf,StiefelGradientAscent(50,0.8,0.5,1.0))
            @test Ur ≈ Mf.U
        end
        
        # Test λ updates
        @testset "updateλ! (RootFinding): t=$t" for t in 1:T
            λr = Ref.updateλ_roots(MM[t].U,MM[t].v,Yf)
            Mf = updateλ!(deepcopy(MM[t]),Yf,RootFinding())
            @test λr ≈ Mf.λ
        end
        
        # Test F/gradF
        @testset "F/gradF: t=$t" for t in 1:T
            Fr = Ref.F(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Ff = HeteroscedasticPCA.F(MM[t].U,MM[t].λ,MM[t].v,Yf)
            @test Fr ≈ Ff
            
            Gr = Ref.gradF(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Gf = HeteroscedasticPCA.gradF(MM[t].U,MM[t].λ,MM[t].v,Yf)
            @test Gr ≈ Gf
        end
    end
end

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
                U, θ, _ = svd(Fref[t])
                vref[t] = Ref.updatev_roots(U,θ.^2,Yblock)
                Fref[t+1] = Ref.updateF_em(Fref[t],vref[t],Yblock)
            end
            U, θ, _ = svd(Fref[end])
            vref[end] = Ref.updatev_roots(U,θ.^2,Yblock)
            
            Fsvd = svd.(Fref)
            @testset "updateF! (ExpectationMaximization)" begin
                for t in 2:length(Fref)
                    M = HPPCA(
                        copy(Fsvd[t-1].U),
                        copy(Fsvd[t-1].S.^2),
                        copy(Fsvd[t-1].Vt),
                        copy(vref[t-1])
                    )
                    updateF!(M,Yblock,ExpectationMaximization())
                    @test M.U ≈ Fsvd[t].U
                    @test M.λ ≈ Fsvd[t].S.^2
                    @test M.Vt ≈ Fsvd[t].Vt
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HPPCA(
                        copy(Fsvd[t].U),
                        copy(Fsvd[t].S.^2),
                        copy(Fsvd[t].Vt),
                        copy(vref[t-1])
                    )
                    updatev!(M,Yblock,RootFinding())
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
                U, θ, _ = svd(Fref[t])
                vref[t] = Ref.updatev_roots(U,θ.^2,Yflatlist)
                Fref[t+1] = Ref.updateF_em(Fref[t],vref[t],Yflatlist)
            end
            U, θ, _ = svd(Fref[end])
            vref[end] = Ref.updatev_roots(U,θ.^2,Yflatlist)
            
            Fsvd = svd.(Fref)
            @testset "updateF! (ExpectationMaximization)" begin
                for t in 2:length(Fref)
                    M = HPPCA(
                        copy(Fsvd[t-1].U),
                        copy(Fsvd[t-1].S.^2),
                        copy(Fsvd[t-1].Vt),
                        copy(vref[t-1])
                    )
                    updateF!(M,Yflatlist,ExpectationMaximization())
                    @test M.U ≈ Fsvd[t].U
                    @test M.λ ≈ Fsvd[t].S.^2
                    @test M.Vt ≈ Fsvd[t].Vt
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HPPCA(
                        copy(Fsvd[t].U),
                        copy(Fsvd[t].S.^2),
                        copy(Fsvd[t].Vt),
                        copy(vref[t-1])
                    )
                    updatev!(M,Yflatlist,RootFinding())
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
        Yflatlist = collect(eachcol(hcat(Y...)))

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
                vref[t] = Ref.updatev_roots(Uref[t],λref[t],Yflatlist)
                λref[t+1] = Ref.updateλ_roots(Uref[t],vref[t],Yflatlist)
                Uref[t+1] = Ref.updateU_mm(Uref[t],λref[t+1],vref[t],Yflatlist)
            end
            vref[end] = Ref.updatev_roots(Uref[end],λref[end],Yflatlist)
            
            VtI = Matrix{Float64}(I,k,k)
            @testset "updateλ! (RootFinding)" begin
                for t in 2:length(λref)
                    M = HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t-1]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    updateλ!(M,Yflatlist,RootFinding())
                    @test M.λ ≈ λref[t]
                end
            end
            @testset "updateU! (MinorizeMaximize)" begin
                for t in 2:length(Uref)
                    M = HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    updateU!(M,Yflatlist,MinorizeMaximize())
                    @test M.U ≈ Uref[t]
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HPPCA(
                        copy(Uref[t]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    updatev!(M,Yflatlist,RootFinding())
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
        Yflatlist = collect(eachcol(hcat(Y...)))

        F0 = randn(rng, d, k)
        iters = 4

        @testset "flat" begin
            Uref = Vector{typeof(F0)}(undef,iters+1)
            λref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            vref = Vector{Vector{eltype(F0)}}(undef,iters+1)
            Q,S,_ = svd(F0)
            Uref[1] = Q[:,1:k]
            λref[1] = S[1:k].^2
            Ynorms = norm.(Yflatlist)
            for t in 1:iters
                vref[t] = Ref.updatev_roots(Uref[t],λref[t],Yflatlist)
                λref[t+1] = Ref.updateλ_roots(Uref[t],vref[t],Yflatlist)
                L = sum(ynorm^2*maximum([λj/vi/(λj+vi) for λj in λref[t+1]]) for (ynorm,vi) in zip(Ynorms,vref[t]))
                Uref[t+1] = Ref.updateU_pga(Uref[t],λref[t+1],vref[t],Yflatlist,1/L)
            end
            vref[end] = Ref.updatev_roots(Uref[end],λref[end],Yflatlist)
            
            VtI = Matrix{Float64}(I,k,k)
            @testset "updateλ! (RootFinding)" begin
                for t in 2:length(λref)
                    M = HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t-1]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    updateλ!(M,Yflatlist,RootFinding())
                    @test M.λ ≈ λref[t]
                end
            end
            @testset "updateU! (ProjectedGradientAscent)" begin
                for t in 2:length(Uref)
                    M = HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    Ynorms = norm.(Yflatlist)
                    L = sum(ynorm^2*maximum([λj/vi/(λj+vi) for λj in M.λ]) for (ynorm,vi) in zip(Ynorms,M.v))
                    updateU!(M,Yflatlist,ProjectedGradientAscent(1/L))
                    @test M.U ≈ Uref[t]
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HPPCA(
                        copy(Uref[t]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    updatev!(M,Yflatlist,RootFinding())
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
        Yflatlist = collect(eachcol(hcat(Y...)))

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
                vref[t] = Ref.updatev_roots(Uref[t],λref[t],Yflatlist)
                λref[t+1] = Ref.updateλ_roots(Uref[t],vref[t],Yflatlist)
                Uref[t+1] = Ref.updateU_sga(Uref[t],λref[t+1],vref[t],Yflatlist,max_line,α,β,σ)
            end
            vref[end] = Ref.updatev_roots(Uref[end],λref[end],Yflatlist)
            
            VtI = Matrix{Float64}(I,k,k)
            @testset "updateλ! (RootFinding)" begin
                for t in 2:length(λref)
                    M = HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t-1]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    updateλ!(M,Yflatlist,RootFinding())
                    @test M.λ ≈ λref[t]
                end
            end
            @testset "updateU! (StiefelGradientAscent)" begin
                for t in 2:length(Uref)
                    M = HPPCA(
                        copy(Uref[t-1]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    updateU!(M,Yflatlist,StiefelGradientAscent(50,0.8,0.5,1.0))
                    @test M.U ≈ Uref[t]
                end
            end
            @testset "updatev! (RootFinding)" begin
                for t in 2:length(vref)
                    M = HPPCA(
                        copy(Uref[t]),
                        copy(λref[t]),
                        copy(VtI),
                        copy(vref[t-1])
                    )
                    updatev!(M,Yflatlist,RootFinding())
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
            Mblock = HPPCA(copy(U), copy(λ), copy(VtI), copy(vv))
            updateλ!(Mblock,Y,RootFinding())
            Mflat = HPPCA(copy(U), copy(λ), copy(VtI), vcat(fill.(copy(vv),n)...))
            updateλ!(Mflat,Yflat,RootFinding())
            Mblock.λ ≈ Mflat.λ
        end
    end
end
