using HePPCAT
using ForwardDiff, LinearAlgebra, StableRNGs, Test

# Internal functions
using HePPCAT: homppca
using HePPCAT: ExpectationMaximization, DifferenceOfConcave,
    MinorizeMaximize, ProjectedGradientAscent, RootFinding, StiefelGradientAscent,
    QuadraticSolvableMinorizer, CubicSolvableMinorizer,
    QuadraticMinorizer, OptimalQuadraticMinorizer
using HePPCAT: ProjectedVariance
using HePPCAT: ArmijoSearch, InverseLipschitz
using HePPCAT: LipBoundU1, LipBoundU2, loglikelihood
using HePPCAT: updateF!, updatev!, updateU!, updateλ!

# Flag to use automatic differentiation
const TEST_WITH_AD = get(ENV,"HEPPCAT_TEST_WITH_AD","false") == "true"

# Load reference implementations
include("ref.jl")

# Convenience functions
factormatrix(M::HePPCATModel) = M.U*sqrt(Diagonal(M.λ))*M.Vt
flatten(M::HePPCATModel,n) = HePPCATModel(M.U,M.λ,M.Vt,vcat(fill.(M.v,n)...))

# Test HePPCATModel type
@testset "HePPCATModel type" begin
    rng = StableRNG(123)
    d, k, v = 10, 3, [4.0,2.0]
    F = randn(rng,d,k)
    U, s, V = svd(F)
    λ = s.^2
    
    @test HePPCATModel(U,λ,V',v) == HePPCATModel(U,λ,V',Int.(v))
    @test HePPCATModel(U,λ,Matrix{Float64}(I,k,k),v) == HePPCATModel(U,λ,I(k),v)
    @test HePPCATModel(U,λ,V',v) == HePPCATModel(F,v)
    @test HePPCATModel(U,λ,V',v).F ≈ F
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
    
    @testset "homoscedastic init" begin
        Ur, λr, vr = Ref.homppca(Yb,k)
        Mb = homppca(Yb,k)
        @test Ur*Diagonal(λr)*Ur' ≈ Mb.U*Diagonal(Mb.λ)*Mb.U'
        @test length(Mb.v) == length(Yb)
        @test all(vr .≈ Mb.v)
        Mf = homppca(Yf,k)
        @test Ur*Diagonal(λr)*Ur' ≈ Mf.U*Diagonal(Mf.λ)*Mf.U'
        @test length(Mf.v) == length(Yf)
        @test all(vr .≈ Mf.v)
    end
    
    @testset "overall function" begin
        Mr = homppca(Yb,k)
        for _ in 1:T
            updatev!(Mr,Yb,ExpectationMaximization())
            updateF!(Mr,Yb,ExpectationMaximization())
        end
        @test Mr !== heppcat(Yb,k,T)
        @test Mr == heppcat(Yb,k,T)
        
        Mr = homppca(Yb,k)
        for _ in 1:T
            updateF!(Mr,Yb,ExpectationMaximization())
        end
        @test Mr !== heppcat(Yb,k,T;vknown=true)
        @test Mr == heppcat(Yb,k,T;vknown=true)
        
        Mr = homppca(Yb,k)
        varfloor = sum(v)/L
        for _ in 1:T
            updatev!(Mr,Yb,ExpectationMaximization())
            Mr.v .= max.(Mr.v,varfloor)
            updateF!(Mr,Yb,ExpectationMaximization())
        end
        @test Mr !== heppcat(Yb,k,T;varfloor=varfloor)
        @test Mr == heppcat(Yb,k,T;varfloor=varfloor)
    end
    
    @testset "block calls" begin
        # Generate sequence of test iterates
        MM = [HePPCATModel(randn(rng,d,k),rand(rng,L))]
        for t in 1:T
            push!(MM, deepcopy(MM[end]))
            updateF!(MM[end],Yb,ExpectationMaximization())
            updatev!(MM[end],Yb,RootFinding())
        end
        
        # Test v updates
        @testset "updatev! (RootFinding): t=$t" for t in 1:T
            vr = Ref.updatev_roots(MM[t].U,MM[t].λ,MM[t].v,Yb)
            Mb = updatev!(deepcopy(MM[t]),Yb,RootFinding())
            @test vr ≈ Mb.v
        end
        @testset "updatev! (ExpectationMaximization): t=$t" for t in 1:T
            vr = Ref.updatev_em(MM[t].U,MM[t].λ,MM[t].v,Yb)
            Mb = updatev!(deepcopy(MM[t]),Yb,ExpectationMaximization())
            @test vr ≈ Mb.v
        end
        @testset "updatev! (DifferenceOfConcave): t=$t" for t in 1:T
            vr = Ref.updatev_doc(MM[t].U,MM[t].λ,MM[t].v,Yb)
            Mb = updatev!(deepcopy(MM[t]),Yb,DifferenceOfConcave())
            @test vr ≈ Mb.v
        end
        @testset "updatev! (QuadraticSolvableMinorizer): t=$t" for t in 1:T
            vr = Ref.updatev_mm_quad(MM[t].U,MM[t].λ,MM[t].v,Yb)
            Mb = updatev!(deepcopy(MM[t]),Yb,QuadraticSolvableMinorizer())
            @test vr ≈ Mb.v
        end
        @testset "updatev! (CubicSolvableMinorizer): t=$t" for t in 1:T
            vr = Ref.updatev_mm_cubic(MM[t].U,MM[t].λ,MM[t].v,Yb)
            Mb = updatev!(deepcopy(MM[t]),Yb,CubicSolvableMinorizer())
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
        @testset "updateU! (ProjectedGradientAscent, Armijo): t=$t" for t in 1:T
            Ur = Ref.updateU_pga_armijo(MM[t].U,MM[t].λ,MM[t].v,Yb,50,0.8,0.5,1e-4)
            Mb = updateU!(deepcopy(MM[t]),Yb,ProjectedGradientAscent(ArmijoSearch(50,0.8,0.5,1e-4)))
            @test Ur ≈ Mb.U
            Mf = updateU!(flatten(deepcopy(MM[t]),n[1:L]),Yf,ProjectedGradientAscent(ArmijoSearch(50,0.8,0.5,1e-4)))
            @test Ur ≈ Mf.U
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
        @testset "updateλ! (ExpectationMaximization): t=$t" for t in 1:T
            λr = Ref.updateλ_em(MM[t].λ,MM[t].U,MM[t].v,Yb)
            Mb = updateλ!(deepcopy(MM[t]),Yb,ExpectationMaximization())
            @test λr ≈ Mb.λ
            Mf = updateλ!(flatten(deepcopy(MM[t]),n[1:L]),Yf,ExpectationMaximization())
            @test λr ≈ Mf.λ
        end
        @testset "updateλ! (MinorizeMaximize): t=$t" for t in 1:T
            λr = Ref.updateλ_mm(MM[t].λ,MM[t].U,MM[t].v,Yb)
            Mb = updateλ!(deepcopy(MM[t]),Yb,MinorizeMaximize())
            @test λr ≈ Mb.λ
            Mf = updateλ!(flatten(deepcopy(MM[t]),n[1:L]),Yf,MinorizeMaximize())
            @test λr ≈ Mf.λ
        end
        @testset "updateλ! (QuadraticMinorizer): t=$t" for t in 1:T
            λr = Ref.updateλ_quad(MM[t].λ,MM[t].U,MM[t].v,Yb)
            Mb = updateλ!(deepcopy(MM[t]),Yb,QuadraticMinorizer())
            @test λr ≈ Mb.λ
            Mf = updateλ!(flatten(deepcopy(MM[t]),n[1:L]),Yf,QuadraticMinorizer())
            @test λr ≈ Mf.λ
        end
        @testset "updateλ! (DifferenceOfConcave): t=$t" for t in 1:T
            λr = Ref.updateλ_doc(MM[t].λ,MM[t].U,MM[t].v,Yb)
            Mb = updateλ!(deepcopy(MM[t]),Yb,DifferenceOfConcave())
            @test λr ≈ Mb.λ
            Mf = updateλ!(flatten(deepcopy(MM[t]),n[1:L]),Yf,DifferenceOfConcave())
            @test λr ≈ Mf.λ
        end
        @testset "updateλ! (OptimalQuadraticMinorizer): t=$t" for t in 1:T
            λr = Ref.updateλ_opt_quad(MM[t].λ,MM[t].U,MM[t].v,Yb)
            Mb = updateλ!(deepcopy(MM[t]),Yb,OptimalQuadraticMinorizer())
            @test λr ≈ Mb.λ
            # Mf = updateλ!(flatten(deepcopy(MM[t]),n[1:L]),Yf,OptimalQuadraticMinorizer())  # Optimal curvature depends on
            # @test λr ≈ Mf.λ                                                                # how the blocks are done
        end
        
        # Test F/gradF
        @testset "F/gradF: t=$t" for t in 1:T
            vf = vcat(fill.(MM[t].v,n[1:L])...)
            Fr = Ref.F(MM[t].U,MM[t].λ,vf,Yf)
            Fb = HePPCAT.F(MM[t].U,MM[t].λ,MM[t].v,Yb)
            @test Fr ≈ Fb
            Ff = HePPCAT.F(MM[t].U,MM[t].λ,vf,Yf)
            @test Fr ≈ Ff

            LLr = [Ref.loglikelihood(U,MM[t].λ,vf,Yf) for U in getfield.(MM,:U)]
            Fb = [HePPCAT.F(U,MM[t].λ,MM[t].v,Yb) for U in getfield.(MM,:U)]
            @test LLr ≈ Fb .+ (LLr[1] - Fb[1])    # "values match up to a constant w.r.t U"
            
            Gr = Ref.gradF(MM[t].U,MM[t].λ,vf,Yf)
            Gb = HePPCAT.gradF(MM[t].U,MM[t].λ,MM[t].v,Yb)
            @test Gr ≈ Gb
            Gf = HePPCAT.gradF(MM[t].U,MM[t].λ,vf,Yf)
            @test Gr ≈ Gf

            if TEST_WITH_AD
                GadLL = ForwardDiff.gradient(U -> Ref.loglikelihood(U,MM[t].λ,vf,Yf),MM[t].U)
                @test GadLL ≈ Gb
                GadF = ForwardDiff.gradient(U -> Ref.F(U,MM[t].λ,vf,Yf),MM[t].U)
                @test GadF ≈ Gb
            end
        end
        
        # Test Lipschitz bound 1 w.r.t U
        @testset "LipBoundU1: t=$t" for t in 1:T
            vf = vcat(fill.(MM[t].v,n[1:L])...)
            Lipr = Ref.LipBoundU1(MM[t].λ,vf,Yf)
            Lipb = LipBoundU1(MM[t],Yb)
            @test Lipr ≈ Lipb
            Lipf = LipBoundU1(flatten(MM[t],n[1:L]),Yf)
            @test Lipr ≈ Lipf

            if TEST_WITH_AD
                Hess = ForwardDiff.hessian(U -> Ref.F(U,MM[t].λ,vf,Yf),MM[t].U)
                @test (1-1e-12)*opnorm(Hess) <= Lipb
            end
        end
        
        # Test Lipschitz bound 2 w.r.t U
        @testset "LipBoundU2: t=$t" for t in 1:T
            Lipr = Ref.LipBoundU2(MM[t].λ,MM[t].v,Yb)
            Lipb = LipBoundU2(MM[t],Yb)
            @test Lipr ≈ Lipb

            if TEST_WITH_AD
                Hess = ForwardDiff.hessian(U -> Ref.F(U,MM[t].λ,MM[t].v,Yb),MM[t].U)
                @test (1-1e-12)*opnorm(Hess) <= Lipb
            end
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
        MM = [HePPCATModel(randn(rng,d,k),rand(rng,sum(n[1:L])))]
        for t in 1:T
            push!(MM, deepcopy(MM[end]))
            updateF!(MM[end],Yf,ExpectationMaximization())
            updatev!(MM[end],Yf,RootFinding())
        end
        
        # Test v updates
        @testset "updatev! (RootFinding): t=$t" for t in 1:T
            vr = Ref.updatev_roots(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Mf = updatev!(deepcopy(MM[t]),Yf,RootFinding())
            @test vr ≈ Mf.v
        end
        @testset "updatev! (ExpectationMaximization): t=$t" for t in 1:T
            vr = Ref.updatev_em(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Mf = updatev!(deepcopy(MM[t]),Yf,ExpectationMaximization())
            @test vr ≈ Mf.v
        end
        @testset "updatev! (DifferenceOfConcave): t=$t" for t in 1:T
            vr = Ref.updatev_doc(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Mf = updatev!(deepcopy(MM[t]),Yf,DifferenceOfConcave())
            @test vr ≈ Mf.v
        end
        @testset "updatev! (QuadraticSolvableMinorizer): t=$t" for t in 1:T
            vr = Ref.updatev_mm_quad(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Mf = updatev!(deepcopy(MM[t]),Yf,QuadraticSolvableMinorizer())
            @test vr ≈ Mf.v
        end
        @testset "updatev! (CubicSolvableMinorizer): t=$t" for t in 1:T
            vr = Ref.updatev_mm_cubic(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Mf = updatev!(deepcopy(MM[t]),Yf,CubicSolvableMinorizer())
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
        @testset "updateU! (ProjectedGradientAscent, Armijo): t=$t" for t in 1:T
            Ur = Ref.updateU_pga_armijo(MM[t].U,MM[t].λ,MM[t].v,Yf,100,0.8,0.5,1e-4)
            Mf = updateU!(deepcopy(MM[t]),Yf,ProjectedGradientAscent(ArmijoSearch(100,0.8,0.5,1e-4)))
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
        @testset "updateλ! (ExpectationMaximization): t=$t" for t in 1:T
            λr = Ref.updateλ_em(MM[t].λ,MM[t].U,MM[t].v,Yf)
            Mf = updateλ!(deepcopy(MM[t]),Yf,ExpectationMaximization())
            @test λr ≈ Mf.λ
        end
        @testset "updateλ! (MinorizeMaximize): t=$t" for t in 1:T
            λr = Ref.updateλ_mm(MM[t].λ,MM[t].U,MM[t].v,Yf)
            Mf = updateλ!(deepcopy(MM[t]),Yf,MinorizeMaximize())
            @test λr ≈ Mf.λ
        end
        @testset "updateλ! (QuadraticMinorizer): t=$t" for t in 1:T
            λr = Ref.updateλ_quad(MM[t].λ,MM[t].U,MM[t].v,Yf)
            Mf = updateλ!(deepcopy(MM[t]),Yf,QuadraticMinorizer())
            @test λr ≈ Mf.λ
        end
        @testset "updateλ! (DifferenceOfConcave): t=$t" for t in 1:T
            λr = Ref.updateλ_doc(MM[t].λ,MM[t].U,MM[t].v,Yf)
            Mf = updateλ!(deepcopy(MM[t]),Yf,DifferenceOfConcave())
            @test λr ≈ Mf.λ
        end
        @testset "updateλ! (OptimalQuadraticMinorizer): t=$t" for t in 1:T
            λr = Ref.updateλ_opt_quad(MM[t].λ,MM[t].U,MM[t].v,Yf)
            Mf = updateλ!(deepcopy(MM[t]),Yf,OptimalQuadraticMinorizer())
            @test λr ≈ Mf.λ
        end
        
        # Test F/gradF
        @testset "F/gradF: t=$t" for t in 1:T
            Fr = Ref.F(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Ff = HePPCAT.F(MM[t].U,MM[t].λ,MM[t].v,Yf)
            @test Fr ≈ Ff

            LLr = [Ref.loglikelihood(U,MM[t].λ,MM[t].v,Yf) for U in getfield.(MM,:U)]
            Ff = [HePPCAT.F(U,MM[t].λ,MM[t].v,Yf) for U in getfield.(MM,:U)]
            @test LLr ≈ Ff .+ (LLr[1] - Ff[1])    # "values match up to a constant w.r.t U"
            
            Gr = Ref.gradF(MM[t].U,MM[t].λ,MM[t].v,Yf)
            Gf = HePPCAT.gradF(MM[t].U,MM[t].λ,MM[t].v,Yf)
            @test Gr ≈ Gf

            if TEST_WITH_AD
                GadLL = ForwardDiff.gradient(U -> Ref.loglikelihood(U,MM[t].λ,MM[t].v,Yf),MM[t].U)
                @test GadLL ≈ Gf
                GadF = ForwardDiff.gradient(U -> Ref.F(U,MM[t].λ,MM[t].v,Yf),MM[t].U)
                @test GadF ≈ Gf
            end
        end
        
        # Test Lipschitz bound 1 w.r.t U
        @testset "LipBoundU1: t=$t" for t in 1:T
            Lipr = Ref.LipBoundU1(MM[t].λ,MM[t].v,Yf)
            Lipf = LipBoundU1(MM[t],Yf)
            @test Lipr ≈ Lipf

            if TEST_WITH_AD
                Hess = ForwardDiff.hessian(U -> Ref.F(U,MM[t].λ,MM[t].v,Yf),MM[t].U)
                @test (1-1e-12)*opnorm(Hess) <= Lipf
            end
        end
        
        # Test Lipschitz bound 2 w.r.t U
        @testset "LipBoundU2: t=$t" for t in 1:T
            Lipr = Ref.LipBoundU2(MM[t].λ,MM[t].v,Yf)
            Lipf = LipBoundU2(MM[t],Yf)
            @test Lipr ≈ Lipf

            if TEST_WITH_AD
                Hess = ForwardDiff.hessian(U -> Ref.F(U,MM[t].λ,MM[t].v,Yf),MM[t].U)
                @test (1-1e-12)*opnorm(Hess) <= Lipf
            end
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
        A = HePPCAT.skew(randn(rng,n,n))
        @test A == HePPCAT.skew(A)
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
    H = HePPCATModel(svd(randn(rng,d,k)).U,λ,Matrix{Float64}(I,k,k),v)
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
    H = HePPCATModel(svd(randn(rng,d,k)).U,λ,Matrix{Float64}(I,k,k),v)
    @test_logs (:warn, "Exceeded maximum line search iterations. Accuracy not guaranteed.") updateU!(deepcopy(H),Y,StiefelGradientAscent(ArmijoSearch(0,0.15,0.5,0.005)))
    @test_logs (:warn, "Exceeded maximum line search iterations. Accuracy not guaranteed.") updateU!(deepcopy(H),Y,StiefelGradientAscent(ArmijoSearch(2,10.0,0.5,0.005)))
end

# Test ProjectedVariance
@testset "ProjectedVariance" begin
    rng = StableRNG(123)
    d, λ, n, v = 10, [4.,2.], [20,20], [1,4]
    k, L = length(λ), length(v)
    F, Z = randn(rng,d,k), [randn(rng,k,n[l]) for l in 1:L]
    Yb = [F*Z[l] + sqrt(v[l])*randn(rng,d,n[l]) for l in 1:L]
    Mb = HePPCATModel(F, fill(sum(v)/L,L))

    vmethods = [
        RootFinding(),
        ExpectationMaximization(),
        DifferenceOfConcave(),
        QuadraticSolvableMinorizer(),
        CubicSolvableMinorizer(),
    ]
    @testset "method=$(method)" for method = vmethods
        Mr = updatev!(deepcopy(Mb),Yb,method)
        @test Mr.v[1] < Mr.v[2]

        # Projection with varfloor < Mr.v1 < Mr.v2
        varfloor = 0.5*Mr.v[1]
        Mp = updatev!(deepcopy(Mb),Yb,ProjectedVariance(method,varfloor))
        @test Mp.v == Mr.v

        # Projection with Mr.v1 < varfloor < Mr.v2
        varfloor = sum(Mr.v)/L
        Mp = updatev!(deepcopy(Mb),Yb,ProjectedVariance(method,varfloor))
        @test Mp.v == [varfloor, Mr.v[2]]

        # Projection with Mr.v1 < Mr.v2 < varfloor
        varfloor = 2*Mr.v[2]
        Mp = updatev!(deepcopy(Mb),Yb,ProjectedVariance(method,varfloor))
        @test Mp.v == [varfloor, varfloor]
    end
end
