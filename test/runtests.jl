using HeteroscedasticPCA
using ForwardDiff, LinearAlgebra, StableRNGs, Test

# Relevant functions
using HeteroscedasticPCA: HPPCA
using HeteroscedasticPCA: ExpectationMaximization, MinorizeMaximize,
    ProjectedGradientAscent, RootFinding, StiefelGradientAscent
using HeteroscedasticPCA: ArmijoSearch, InverseLipschitz
using HeteroscedasticPCA: LipBoundU1, LipBoundU2, loglikelihood
using HeteroscedasticPCA: updateF!, updatev!, updateU!, updateλ!

# Load reference implementations
include("ref.jl")

# Convenience functions
factormatrix(M::HPPCA) = M.U*sqrt(Diagonal(M.λ))*M.Vt
flatten(M::HPPCA,n) = HPPCA(M.U,M.λ,M.Vt,vcat(fill.(M.v,n)...))

# Test convenience constructors and equality
@testset "Constructors" begin
    rng = StableRNG(123)
    d, k, v = 10, 3, [4.0,2.0]
    F = randn(rng,d,k)
    U, s, V = svd(F)
    λ = s.^2
    
    @test HPPCA(U,λ,V',v) == HPPCA(U,λ,V',Int.(v))
    @test HPPCA(U,λ,Matrix{Float64}(I,k,k),v) == HPPCA(U,λ,I(k),v)
    @test HPPCA(U,λ,V',v) == HPPCA(F,v)
end

# Test all updates
T = 5
n, v = (40, 10), (4, 1)
@testset "n=$(n[1:L]), v=$(v[1:L]), d=$d, k=$k" for L = 1:2, d = 5:10:25, k = 1:3
    # Compute test problem
    rng = StableRNG(123)
    F, Z = randn(rng,d,k), [randn(rng,k,n[l]) for l in 1:L]
    Yb = [F*Z[l] + sqrt(v[l])*randn(rng,d,n[l]) for l in 1:L]   # blocked
    Yf = reshape.(collect.(eachcol(hcat(Yb...))),:,1)           # flatten
    
    @testset "block calls" begin
        # Generate sequence of test iterates
        MM = [HPPCA(randn(rng,d,k),rand(rng,L))]
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
        @testset "updatev! (ExpectationMaximization): t=$t" for t in 1:T
            vr = Ref.updatev_em(MM[t].U,MM[t].λ,MM[t].v,Yb)
            Mb = updatev!(deepcopy(MM[t]),Yb,ExpectationMaximization())
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
        @testset "updateU! (ProjectedGradientAscent, $stepsize): t=$t" for stepsize in [0.0,0.5,Inf], t in 1:T
            Ur = Ref.updateU_pga(MM[t].U,MM[t].λ,MM[t].v,Yb,stepsize)
            Mb = updateU!(deepcopy(MM[t]),Yb,ProjectedGradientAscent(stepsize))
            @test Ur ≈ Mb.U
            Mf = updateU!(flatten(deepcopy(MM[t]),n[1:L]),Yf,ProjectedGradientAscent(stepsize))
            @test Ur ≈ Mf.U
        end
        @testset "updateU! (ProjectedGradientAscent, Lipschitz): t=$t" for t in 1:T
            Lip = Ref.LipBoundU2(MM[t].λ,MM[t].v,Yb)
            Ur = Ref.updateU_pga(MM[t].U,MM[t].λ,MM[t].v,Yb,1/Lip)
            Mb = updateU!(deepcopy(MM[t]),Yb,ProjectedGradientAscent(InverseLipschitz()))
            @test Ur ≈ Mb.U
            # Mf = updateU!(flatten(deepcopy(MM[t]),n[1:L]),Yf,ProjectedGradientAscent(InverseLipschitz()))  # LipBound2 depends on
            # @test Ur ≈ Mf.U                                                                                # how the blocks are done
        end
        @testset "updateU! (StiefelGradientAscent): t=$t" for t in 1:T
            Ur = Ref.updateU_sga(MM[t].U,MM[t].λ,MM[t].v,Yb,50,0.8,0.5,0.5)
            Mb = updateU!(deepcopy(MM[t]),Yb,StiefelGradientAscent(ArmijoSearch(50,0.8,0.5,0.5)))
            @test Ur ≈ Mb.U
            Mf = updateU!(flatten(deepcopy(MM[t]),n[1:L]),Yf,StiefelGradientAscent(ArmijoSearch(50,0.8,0.5,0.5)))
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

            LLr = [Ref.loglikelihood(U,MM[t].λ,vf,Yf) for U in getfield.(MM,:U)]
            Fb = [HeteroscedasticPCA.F(U,MM[t].λ,MM[t].v,Yb) for U in getfield.(MM,:U)]
            @test LLr ≈ Fb .+ (LLr[1] - Fb[1])    # "values match up to a constant w.r.t U"
            
            Gr = Ref.gradF(MM[t].U,MM[t].λ,vf,Yf)
            Gb = HeteroscedasticPCA.gradF(MM[t].U,MM[t].λ,MM[t].v,Yb)
            @test Gr ≈ Gb
            Gf = HeteroscedasticPCA.gradF(MM[t].U,MM[t].λ,vf,Yf)
            @test Gr ≈ Gf

            GadLL = ForwardDiff.gradient(U -> Ref.loglikelihood(U,MM[t].λ,vf,Yf),MM[t].U)
            @test GadLL ≈ Gb
            GadF = ForwardDiff.gradient(U -> Ref.F(U,MM[t].λ,vf,Yf),MM[t].U)
            @test GadF ≈ Gb
        end
        
        # Test Lipschitz bound 1 w.r.t U
        @testset "LipBoundU1: t=$t" for t in 1:T
            vf = vcat(fill.(MM[t].v,n[1:L])...)
            Lipr = Ref.LipBoundU1(MM[t].λ,vf,Yf)
            Lipb = LipBoundU1(MM[t],Yb)
            @test Lipr ≈ Lipb
            Lipf = LipBoundU1(flatten(MM[t],n[1:L]),Yf)
            @test Lipr ≈ Lipf

            Hess = ForwardDiff.hessian(U -> Ref.F(U,MM[t].λ,vf,Yf),MM[t].U)
            @test (1-1e-12)*opnorm(Hess) <= Lipb
        end
        
        # Test Lipschitz bound 2 w.r.t U
        @testset "LipBoundU2: t=$t" for t in 1:T
            Lipr = Ref.LipBoundU2(MM[t].λ,MM[t].v,Yb)
            Lipb = LipBoundU2(MM[t],Yb)
            @test Lipr ≈ Lipb

            Hess = ForwardDiff.hessian(U -> Ref.F(U,MM[t].λ,MM[t].v,Yb),MM[t].U)
            @test (1-1e-12)*opnorm(Hess) <= Lipb
        end
        
        # Test log-likelihood
        @testset "loglikelihood: t=$t" for t in 1:T
            vf = vcat(fill.(MM[t].v,n[1:L])...)
            LLr = Ref.loglikelihood(MM[t].U,MM[t].λ,vf,Yf)
            LLb = loglikelihood(MM[t],Yb)
            @test LLr ≈ LLb
            LLf = loglikelihood(flatten(MM[t],n[1:L]),Yf)
            @test LLr ≈ LLf
        end
    end
    
    @testset "flat calls" begin
        # Generate sequence of test iterates
        MM = [HPPCA(randn(rng,d,k),rand(rng,sum(n[1:L])))]
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
        @testset "updatev! (ExpectationMaximization): t=$t" for t in 1:T
            vr = Ref.updatev_em(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Mf = updatev!(deepcopy(MM[t]),Yf,ExpectationMaximization())
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
        @testset "updateU! (ProjectedGradientAscent, $stepsize): t=$t" for stepsize in [0.0,0.5,Inf], t in 1:T
            Ur = Ref.updateU_pga(MM[t].U,MM[t].λ,MM[t].v,Yf,stepsize)
            Mf = updateU!(deepcopy(MM[t]),Yf,ProjectedGradientAscent(stepsize))
            @test Ur ≈ Mf.U
        end
        @testset "updateU! (ProjectedGradientAscent, Lipschitz): t=$t" for t in 1:T
            Lip = Ref.LipBoundU2(MM[t].λ,MM[t].v,Yf)
            Ur = Ref.updateU_pga(MM[t].U,MM[t].λ,MM[t].v,Yf,1/Lip)
            Mf = updateU!(deepcopy(MM[t]),Yf,ProjectedGradientAscent(InverseLipschitz()))
            @test Ur ≈ Mf.U
        end
        @testset "updateU! (StiefelGradientAscent): t=$t" for t in 1:T
            Ur = Ref.updateU_sga(MM[t].U,MM[t].λ,MM[t].v,Yf,50,0.8,0.5,0.5)
            Mf = updateU!(deepcopy(MM[t]),Yf,StiefelGradientAscent(ArmijoSearch(50,0.8,0.5,0.5)))
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

            LLr = [Ref.loglikelihood(U,MM[t].λ,MM[t].v,Yf) for U in getfield.(MM,:U)]
            Ff = [HeteroscedasticPCA.F(U,MM[t].λ,MM[t].v,Yf) for U in getfield.(MM,:U)]
            @test LLr ≈ Ff .+ (LLr[1] - Ff[1])    # "values match up to a constant w.r.t U"
            
            Gr = Ref.gradF(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Gf = HeteroscedasticPCA.gradF(MM[t].U,MM[t].λ,MM[t].v,Yf)
            @test Gr ≈ Gf

            GadLL = ForwardDiff.gradient(U -> Ref.loglikelihood(U,MM[t].λ,MM[t].v,Yf),MM[t].U)
            @test GadLL ≈ Gf
            GadF = ForwardDiff.gradient(U -> Ref.F(U,MM[t].λ,MM[t].v,Yf),MM[t].U)
            @test GadF ≈ Gf
        end
        
        # Test Lipschitz bound 1 w.r.t U
        @testset "LipBoundU1: t=$t" for t in 1:T
            Lipr = Ref.LipBoundU1(MM[t].λ,MM[t].v,Yf)
            Lipf = LipBoundU1(MM[t],Yf)
            @test Lipr ≈ Lipf

            Hess = ForwardDiff.hessian(U -> Ref.F(U,MM[t].λ,MM[t].v,Yf),MM[t].U)
            @test (1-1e-12)*opnorm(Hess) <= Lipf
        end
        
        # Test Lipschitz bound 2 w.r.t U
        @testset "LipBoundU2: t=$t" for t in 1:T
            Lipr = Ref.LipBoundU2(MM[t].λ,MM[t].v,Yf)
            Lipf = LipBoundU2(MM[t],Yf)
            @test Lipr ≈ Lipf

            Hess = ForwardDiff.hessian(U -> Ref.F(U,MM[t].λ,MM[t].v,Yf),MM[t].U)
            @test (1-1e-12)*opnorm(Hess) <= Lipf
        end

        # Test log-likelihood
        @testset "loglikelihood: t=$t" for t in 1:T
            LLr = Ref.loglikelihood(MM[t].U,MM[t].λ,MM[t].v,Yf)
            LLf = loglikelihood(MM[t],Yf)
            @test LLr ≈ LLf
        end
    end
end

# Test skew-symmetry on several matrices
@testset "skew" begin
    rng = StableRNG(123)
    @testset "random $n x $n matrix" for n in 1:20
        A = HeteroscedasticPCA.skew(randn(rng,n,n))
        @test A == HeteroscedasticPCA.skew(A)
        @test A == -A'
    end
end

# Test orthonormality of StiefelGradientAscent updates
@testset "StiefelGradientAscent orthonormality" begin
    rng = StableRNG(123)
    d, λ, n, v = 100, [4.,2.], [200,200], [0.2,0.4]
    k = length(λ)
    U = svd(randn(rng,d,k)).U
    F = U*sqrt(Diagonal(λ))
    Y = [F*randn(rng,k,nl) + sqrt(vl)*randn(rng,d,nl) for (nl,vl) in zip(n,v)]
    H = HPPCA(svd(randn(rng,d,k)).U,λ,Matrix{Float64}(I,k,k),v)
    @testset "iterate $t" for t in 1:200
        updateU!(H,Y,StiefelGradientAscent(ArmijoSearch(10,0.15,0.5,0.005)))
        @test H.U'*H.U ≈ I
    end
end

# Test StiefelGradientAscent line search warning
@testset "StiefelGradientAscent line search warning" begin
    rng = StableRNG(123)
    d, λ, n, v = 100, [4.,2.], [200,200], [0.2,0.4]
    k = length(λ)
    U = svd(randn(rng,d,k)).U
    F = U*sqrt(Diagonal(λ))
    Y = [F*randn(rng,k,nl) + sqrt(vl)*randn(rng,d,nl) for (nl,vl) in zip(n,v)]
    H = HPPCA(svd(randn(rng,d,k)).U,λ,Matrix{Float64}(I,k,k),v)
    @test_logs (:warn, "Exceeded maximum line search iterations. Accuracy not guaranteed.") updateU!(deepcopy(H),Y,StiefelGradientAscent(ArmijoSearch(0,0.15,0.5,0.005)))
    @test_logs (:warn, "Exceeded maximum line search iterations. Accuracy not guaranteed.") updateU!(deepcopy(H),Y,StiefelGradientAscent(ArmijoSearch(2,10.0,0.5,0.005)))
end
