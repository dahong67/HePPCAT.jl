using HeteroscedasticPCA
using LinearAlgebra, Random, Test

# Relevant functions
using HeteroscedasticPCA: HPPCA
using HeteroscedasticPCA: ExpectationMaximization, MinorizeMaximize,
    ProjectedGradientAscent, RootFinding, StiefelGradientAscent
using HeteroscedasticPCA: LipBoundU, loglikelihood
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
            Lip = Ref.LipBoundU(MM[t].λ,MM[t].v,Yb)
            Ur = Ref.updateU_pga(MM[t].U,MM[t].λ,MM[t].v,Yb,1/Lip)
            Mb = updateU!(deepcopy(MM[t]),Yb,ProjectedGradientAscent(1/LipBoundU(MM[t],norm.(Yb))))
            @test Ur ≈ Mb.U
            Mf = updateU!(flatten(deepcopy(MM[t]),n[1:L]),Yf,ProjectedGradientAscent(1/LipBoundU(flatten(MM[t],n[1:L]),norm.(Yf))))
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
        
        # Test Lipschitz bound w.r.t U
        @testset "LipBoundU: t=$t" for t in 1:T
            vf = vcat(fill.(MM[t].v,n[1:L])...)
            Lipr = Ref.LipBoundU(MM[t].λ,vf,Yf)
            Lipb = LipBoundU(MM[t],norm.(Yb))
            @test Lipr ≈ Lipb
            Lipf = LipBoundU(flatten(MM[t],n[1:L]),norm.(Yf))
            @test Lipr ≈ Lipf
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
            Lip = Ref.LipBoundU(MM[t].λ,MM[t].v,Yf)
            Ur = Ref.updateU_pga(MM[t].U,MM[t].λ,MM[t].v,Yf,1/Lip)
            Mf = updateU!(deepcopy(MM[t]),Yf,ProjectedGradientAscent(1/LipBoundU(MM[t],norm.(Yf))))
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
        
        # Test Lipschitz bound w.r.t U
        @testset "LipBoundU: t=$t" for t in 1:T
            Lipr = Ref.LipBoundU(MM[t].λ,MM[t].v,Yf)
            Lipf = LipBoundU(MM[t],norm.(Yf))
            @test Lipr ≈ Lipf
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
rng = MersenneTwister(123)
@testset "skew: random $n x $n matrix" for n in 1:20
    A = HeteroscedasticPCA.skew(randn(rng,n,n))
    @test HeteroscedasticPCA.skew(A) == A == -A'
end