module Ref

using IdentityRanges: IdentityRange
using IntervalArithmetic: interval, mid
using IntervalRootFinding: Newton, roots
using LinearAlgebra: Diagonal, Hermitian, I, eigen, opnorm, norm, qr, svd, tr, /
using Logging: @debug
import PolynomialRoots
using Roots: find_zero
using Statistics: mean

# findmax from https://github.com/cmcaine/julia/blob/argmax-2-arg-harder/base/reduce.jl#L704-L705
# argmax from https://github.com/cmcaine/julia/blob/argmax-2-arg-harder/base/reduce.jl#L830
# part of pull request https://github.com/JuliaLang/julia/pull/35316
_findmax(f, domain) = mapfoldl(x -> (f(x), x), _rf_findmax, domain)
_rf_findmax((fm, m), (fx, x)) = isless(fm, fx) ? (fx, x) : (fm, m)
_argmax(f, domain) = _findmax(f, domain)[2]

# Homoscedastic init
function homppca(Y,k)
    Yflat = reduce(hcat,Y)
    d, n = size(Yflat)
    λh, Uh = eigen(Hermitian(Yflat*Yflat'/n),sortby=-)
    λb = mean(λh[k+1:d])
    return Uh[:,1:k], λh[1:k] .- λb, λb
end

# log-likelihood (todo: add constant)
function loglikelihood(U,λ,v,Y)
    d, k = size(U)
    n, L = size.(Y,2), length(Y)
    return 1/2*sum(
        -n[l]*sum(log.(λ .+ v[l])) - n[l]*(d-k)*log(v[l]) - norm(Y[l])^2/v[l]
        + norm(sqrt(Diagonal((λ./v[l])./(λ .+ v[l])))*U'Y[l])^2
        for l = 1:L
    )
end

# v updates
updatev_roots(U,λ,v,Y) = [updatevl_roots(U,λ,v[l],Y[l]) for l in 1:length(Y)]
function updatevl_roots(U,λ,vl,Yl)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients and check edge case
    α = [j == 0 ? d-k : 1 for j in IdentityRange(0:k)]
    β = [j == 0 ? norm((I-U*U')*Yl)^2/nl : norm(Yl'U[:,j])^2/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    J0 = findall(iszero,γ)
    βtl = sum(β[j] for j in J0)
    iszero(βtl) && return zero(vl)

    # Find nonnegative critical points
    tol = 1e-8  # todo: choose tolerance adaptively
    vmin, vmax = extrema(β[j]/α[j]-γ[j] for j in 0:k if !(iszero(α[j]) && iszero(β[j])))
    vcritical = roots(interval(vmin,vmax) ∩ interval(zero(vl),Inf),Newton,tol) do v
        - sum(α[j]/(γ[j]+v) for j in 0:k if !iszero(α[j])) + sum(β[j]/(γ[j]+v)^2 for j in 0:k if !iszero(β[j]))
    end

    # Return maximizer
    return _argmax(mid.(interval.(vcritical))) do v
        -(sum(α[j]*log(γ[j]+v) for j in 0:k if !iszero(α[j])) + sum(β[j]/(γ[j]+v) for j in 0:k if !iszero(β[j])))
    end
end
function updatev_em(U,λ,v,Y)
    d, n, L = size(U,1), size.(Y,2), length(Y)
    Λ = Diagonal(λ)
    ρ = [norm((I-U*Λ*inv(Λ+v[l]*I)*U')*Y[l])^2/n[l] + v[l]*tr(U*Λ*inv(Λ+v[l]*I)*U') for l in 1:L]
    return ρ./d
end
updatev_doc(U,λ,v,Y) = [updatevl_doc(U,λ,v[l],Y[l]) for l in 1:length(Y)]
function updatevl_doc(U,λ,vl,Yl)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients and check edge case
    α = [j == 0 ? d-k : 1 for j in IdentityRange(0:k)]
    β = [j == 0 ? norm((I-U*U')*Yl)^2/nl : norm(Yl'U[:,j])^2/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    affslope = -sum(α[j]/(γ[j]+vl) for j in 0:k if !iszero(α[j]))
    Ltp = v -> affslope + sum(β[j]/(γ[j]+v)^2 for j in 0:k if !iszero(β[j]))
    if affslope == -Inf || Ltp(zero(vl)) <= zero(vl)
        return zero(vl)
    end

    # Return nonnegative critical point
    tol = 1e-8  # todo: choose tolerance adaptively
    vmax = maximum(sqrt(β[j]/α[j]*(γ[j]+vl)) - γ[j] for j in 0:k if !(iszero(α[j]) && iszero(β[j])))
    return find_zero(Ltp, (zero(vmax),vmax))
end
updatev_mm_quad(U,λ,v,Y) = [updatevl_mm_quad(U,λ,v[l],Y[l]) for l in 1:length(Y)]
function updatevl_mm_quad(U,λ,vl,Yl)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients
    α = [j == 0 ? d-k : 1 for j in IdentityRange(0:k)]
    β = [j == 0 ? norm((I-U*U')*Yl)^2/nl : norm(Yl'U[:,j])^2/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    J0 = findall(iszero,γ)
    αtl = sum(α[j] for j in J0)
    βtl = sum(β[j] for j in J0)
    ζtl = sum(α[j]/(γ[j]+vl) for j in 0:k if j ∉ J0)
    B = βtl + sum(β[j]*vl^2/(γ[j]+vl)^2 for j in 0:k if j ∉ J0)

    return (-αtl + sqrt(αtl^2 + 4*ζtl*B))/(2*ζtl)
end
updatev_mm_cubic(U,λ,v,Y) = [updatevl_mm_cubic(U,λ,v[l],Y[l]) for l in 1:length(Y)]
function updatevl_mm_cubic(U,λ,vl,Yl)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients
    α = [j == 0 ? d-k : 1 for j in IdentityRange(0:k)]
    β = [j == 0 ? norm((I-U*U')*Yl)^2/nl : norm(Yl'U[:,j])^2/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    c = [-2*β[j]/γ[j]^3 for j in IdentityRange(0:k)]
    J0 = findall(iszero,γ)
    αtl = sum(α[j] for j in J0)
    βtl = sum(β[j] for j in J0)
    ζtl = sum(α[j]/(γ[j]+vl) for j in 0:k if j ∉ J0)
    γtl = -ζtl + sum(β[j]/(γ[j]+vl)^2 for j in 0:k if j ∉ J0)
    ctl = sum(c[j] for j in 0:k if j ∉ J0)

    complexroots = PolynomialRoots.solve_cubic_eq(complex.([βtl,-αtl,γtl-ctl*vl,ctl]))
    vcritical = [real(v) for v in complexroots if real(v) ≈ v && real(v) >= zero(vl)]

    isempty(vcritical) && return zero(vl)
    return _argmax(vcritical) do v
        -αtl*log(v) - βtl/v - ζtl*v + sum(β[j]/(γ[j]+vl)^2*v + (1/2)*c[j]*(v-vl)^2 for j in 0:k if j ∉ J0)
    end
end

# F updates
function updateF_em(F,v,Y)
    n, L = size.(Y,2), length(Y)
    U, θ, V = svd(F)
    λ = θ.^2
    
    Λ = Diagonal(λ)
    Γ = [inv(Λ + v[l]*I) for l in 1:L]
    Z = [Γ[l]*sqrt(Λ)*U'*Y[l] for l in 1:L]
    num = sum(Y[l]*Z[l]'/v[l] for l in 1:L)
    den = sum(Z[l]*Z[l]'/v[l] + n[l]*Γ[l] for l in 1:L)

    return (num / den) * V'
end

# U updates
function polar(A)
    F = svd(A)
    return F.U*F.Vt
end
gradF(U,λ,v,Y) = sum(Yl * Yl' * U * Diagonal(λ./vl./(λ.+vl)) for (Yl,vl) in zip(Y,v))
F(U,λ,v,Y) = 1/2*sum(norm(sqrt(Diagonal(λ./vl./(λ.+vl)))*U'*Yl)^2 for (Yl,vl) in zip(Y,v))
function LipBoundU1(λ,v,Y)
    L, λmax = length(v), maximum(λ)
    return sum(norm(Y[l])^2*λmax/v[l]/(λmax+v[l]) for l in 1:L)
end
function LipBoundU2(λ,v,Y)
    L = length(v)
    return sum(opnorm(Y[l]*Y[l]') * opnorm(Diagonal(λ./v[l]./(λ.+v[l]))) for l in 1:L)
end

updateU_mm(U,λ,v,Y) = polar(gradF(U,λ,v,Y))
updateU_pga(U,λ,v,Y,α) = α == Inf ? polar(gradF(U,λ,v,Y)) : polar(U + α*gradF(U,λ,v,Y))
function updateU_pga_armijo(U,λ,v,Y,maxsearches,stepsize,contraction,tol)
    dFdU = gradF(U,λ,v,Y)
    F0, FΔ = F(U,λ,v,Y), tol * norm(dFdU)^2
    for m in 0:maxsearches-1
        Δ = stepsize * contraction^m
        (F(polar(U + Δ*dFdU),λ,v,Y) >= F0 + Δ * FΔ) && return polar(U + Δ*dFdU)
    end
    @debug "Exceeded maximum line search iterations. Accuracy not guaranteed."
    Δ = stepsize * contraction^maxsearches
    return polar(U + Δ*dFdU)
end
function updateU_sga(U,λ,v,Y,maxsearches,stepsize,contraction,tol)
    dFdU = gradF(U,λ,v,Y)
    ∇F = dFdU - U*(dFdU'U)
    
    F0, FΔ = F(U,λ,v,Y), tol * norm(∇F)^2
    for m in 0:maxsearches-1
        Δ = stepsize * contraction^m
        (F(geodesic(U,∇F,Δ),λ,v,Y) >= F0 + Δ * FΔ) && return geodesic(U,∇F,Δ)
    end
    @debug "Exceeded maximum line search iterations. Accuracy not guaranteed."
    Δ = stepsize * contraction^maxsearches
    return geodesic(U,∇F,Δ)
end
skew(A) = (A-A')/2
function geodesic(U,X,t)
    k = size(U,2)

    A = skew(U'X)
    Q,R = qr(X - U*(U'X))

    MN = exp(t*[A -R'; R zeros(k,k)])[:,1:k]
    M, N = MN[1:k,:], MN[k+1:end,:]

    return U*M + Matrix(Q)*N
end

# λ updates
ispos(x) = x > zero(x)

updateλ_roots(U,v,Y) = [updateλj_roots(uj,v,Y) for uj in eachcol(U)]
function updateλj_roots(uj,v,Y)
    n, L = size.(Y,2), length(Y)

    # Compute coefficients and root bounds
    β = [norm(uj'Yl)^2 for Yl in Y]
    λmin, λmax = extrema(max.(zero.(β), β./n .- v))

    # Compute roots
    tol = 1e-8  # todo: choose tolerance adaptively
    λmax-λmin < tol && return (λmax+λmin)/2
    λcritical = roots(interval(λmin,λmax),Newton,tol) do λ
        sum(β[l]/(λ+v[l])^2 - n[l]/(λ+v[l]) for l in 1:L)
    end
    isempty(λcritical) && return λmin
    length(λcritical) == 1 && return mid(interval(first(λcritical)))

    # Return maximizer
    return _argmax(mid.(interval.(λcritical))) do λ
        -sum(n[l]*log(λ+v[l]) + β[l]/(λ+v[l]) for l in 1:L)
    end
end
updateλ_em(λ,U,v,Y) = [updateλj_em(λj,uj,v,Y) for (λj,uj) in zip(λ,eachcol(U))]
function updateλj_em(λj,uj,v,Y)
    n, L = size.(Y,2), length(Y)
    θj = sum(sqrt(λj)/v[l]/(λj+v[l])*norm(uj'Y[l])^2 for l = 1:L)/sum(λj/v[l]/(λj+v[l])^2*norm(uj'Y[l])^2 + n[l]/(λj+v[l]) for l = 1:L)
    θj^2
end
updateλ_mm(λ,U,v,Y) = [updateλj_mm(λj,uj,v,Y) for (λj,uj) in zip(λ,eachcol(U))]
function updateλj_mm(λj,uj,v,Y)
    all(ispos,v) || throw("Minorizer expects positive v. Got: $v")
    n, L = size.(Y,2), length(Y)

    ζ = [norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) + n[l] for l in 1:L]
    num = sum(norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) for l in 1:L) * sum(ζ[l]*v[l]/(λj+v[l]) for l in 1:L)
    den = sum(ζ[l]/(λj+v[l]) for l in 1:L)
    return (1/sum(n)) * num / den
end
updateλ_quad(λ,U,v,Y) = [updateλj_quad(λj,uj,v,Y) for (λj,uj) in zip(λ,eachcol(U))]
function updateλj_quad(λj,uj,v,Y)
    all(ispos,v) || throw("Minorizer expects positive v. Got: $v")
    n, L = size.(Y,2), length(Y)
    
    num = sum(1:L) do l
        ytlj = norm(Y[l]'uj)^2
        n[l]/(λj+v[l]) - ytlj/(λj+v[l])^2
    end
    den = sum(1:L) do l
        ytlj = norm(Y[l]'uj)^2
        ctlj = -2*ytlj/v[l]^3
        ctlj
    end
    
    return max(zero(λj),λj + num/den)
end
updateλ_doc(λ,U,v,Y) = [updateλj_doc(λj,uj,v,Y) for (λj,uj) in zip(λ,eachcol(U))]
function updateλj_doc(λj,uj,v,Y)
    n, L = size.(Y,2), length(Y)
    
    # Compute coefficients and check edge case
    ytj = [norm(Y[l]'uj)^2 for l in 1:L]
    affslope = -sum(n[l]/(λj+v[l]) for l in 1:L)
    Ltp = λ -> affslope + sum(ytj[l]/(λ+v[l])^2 for l in 1:L if !iszero(ytj[l]))
    if affslope == -Inf || Ltp(zero(λj)) <= zero(λj)
        return zero(λj)
    end

    # Return nonnegative critical point
    tol = 1e-13  # to get a bracketing interval
    λmax = maximum(sqrt(ytj[l]/n[l]*(λj+v[l])) - v[l] for l in 1:L) + tol
    return find_zero(Ltp,(zero(λmax),λmax))
end
updateλ_opt_quad(λ,U,v,Y) = [updateλj_opt_quad(λj,uj,v,Y) for (λj,uj) in zip(λ,eachcol(U))]
function updateλj_opt_quad(λj,uj,v,Y)
    all(ispos,v) || throw("Minorizer expects positive v. Got: $v")
    n, L = size.(Y,2), length(Y)
    
    num = sum(1:L) do l
        ytlj = norm(Y[l]'uj)^2
        n[l]/(λj+v[l]) - ytlj/(λj+v[l])^2
    end
    den = sum(1:L) do l
        ytlj = norm(Y[l]'uj)^2
        a, b = n[l], ytlj
        f = λ -> a*log(λ+v[l]) + b/(λ+v[l])
        fd = λ -> a/(λ+v[l]) - b/(λ+v[l])^2
        ctlj = (2b <= a*v[l]) ? zero(λj) : -2*(f(zero(λj)) - f(λj) + fd(λj)*λj)/λj^2
        ctlj
    end
    
    return max(zero(λj),λj + num/den)
end

end
