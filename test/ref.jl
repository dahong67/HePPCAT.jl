module Ref

using IntervalArithmetic: interval, mid
using IntervalRootFinding: Newton, roots
using LinearAlgebra: Diagonal, I, norm, qr, svd, tr, /
using Logging: @debug

# findmax from https://github.com/cmcaine/julia/blob/argmax-2-arg-harder/base/reduce.jl#L704-L705
# argmax from https://github.com/cmcaine/julia/blob/argmax-2-arg-harder/base/reduce.jl#L830
# part of pull request https://github.com/JuliaLang/julia/pull/35316
_findmax(f, domain) = mapfoldl(x -> (f(x), x), _rf_findmax, domain)
_rf_findmax((fm, m), (fx, x)) = isless(fm, fx) ? (fx, x) : (fm, m)
_argmax(f, domain) = _findmax(f, domain)[2]

# Types
struct HPPCA{T<:AbstractFloat}
    U::Matrix{T}   # eigvecs of FF' (factor/spike covariance)
    λ::Vector{T}   # eigvals of FF' (factor/spike covariance)
    Vt::Matrix{T}  # (transposed) eigvecs of F'F (i.e., right singvecs of F)
    v::Vector{T}   # block noise variances
    function HPPCA{T}(U,λ,Vt,v) where {T<:AbstractFloat}
        size(U,2) == length(λ) || throw(DimensionMismatch("U has dimensions $(size(U)) but λ has length $(length(λ))"))
        size(Vt,1) == length(λ) || throw(DimensionMismatch("Vt has dimensions $(size(Vt)) but λ has length $(length(λ))"))
        new{T}(U,λ,Vt,v)
    end
end
HPPCA(U::Matrix{T},λ::Vector{T},Vt::Matrix{T},v::Vector{T}) where {T<:AbstractFloat} = HPPCA{T}(U,λ,Vt,v)

struct RootFinding end
struct ExpectationMaximization end
struct MinorizeMaximize end
struct ProjectedGradientAscent{T<:AbstractFloat}
    stepsize::T
end
struct StiefelGradientAscent{S<:Integer,T<:AbstractFloat}
    maxsearches::S  # maximum number of line searches
    stepsize::T     # initial stepsize
    contraction::T  # contraction factor
    tol::T          # tolerance for sufficient decrease
end

# PPCA
function ppca(Y,k,iters,init,::Val{:sage})
    M = HPPCA(svd(init).U,svd(init).S.^2,svd(init).Vt,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        updateF!(M,Y,ExpectationMaximization())
        push!(MM, deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM, :λ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:mm})
    M = HPPCA(svd(init).U,svd(init).S.^2,svd(init).Vt,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        updateλ!(M,Y,RootFinding())
        updateU!(M,Y,MinorizeMaximize())
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:λ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:pgd})
    Ynorms = vec(mapslices(norm,hcat(Y...),dims=1))
    M = HPPCA(svd(init).U,svd(init).S.^2,svd(init).Vt,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        updateλ!(M,Y,RootFinding())
        L = sum(ynorm^2*maximum([λj/vi/(λj+vi) for λj in M.λ])
            for (ynorm,vi) in zip(Ynorms,M.v))
        updateU!(M,Y,ProjectedGradientAscent(1/L))
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:λ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:sgd},max_line=50,α=0.8,β=0.5,σ=1.0)
    M = HPPCA(svd(init).U,svd(init).S.^2,svd(init).Vt,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        updateλ!(M,Y,RootFinding())
        updateU!(M,Y,StiefelGradientAscent(max_line,α,β,σ))
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:λ), getfield.(MM,:v)
end

# Updates: F
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

# Updates: v
updatev_roots(U,λ,Y) = [updatevl_roots(U,λ,Yl) for Yl in Y]
function updatevl_roots(U,λ,Yl)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients and root bounds
    α0, β0 = d-k, norm(Yl-U*U'Yl)^2/nl
    β = [norm(uj'Yl)^2/nl for uj in eachcol(U)]
    vmin, vmax = extrema([β0/α0; max.(zero.(β), β .- λ)])

    # Compute roots
    tol = 1e-8  # todo: choose tolerance adaptively
    vmax-vmin < tol && return (vmax+vmin)/2
    vcritical = roots(interval(vmin,vmax),Newton,tol) do v
        β0/v^2-α0/v + sum(β[j]/(λ[j]+v)^2 - 1/(λ[j]+v) for j in 1:k)
    end
    isempty(vcritical) && return vmin
    length(vcritical) == 1 && return mid(interval(only(vcritical)))

    # Return maximizer
    return _argmax(mid.(interval.(vcritical))) do v
        -(α0*log(v)+β0/v + sum(log(λ[j]+v) + β[j]/(λ[j]+v) for j in 1:k))
    end
end

# Updates: U
function polar(A)
    F = svd(A)
    return F.U*F.Vt
end
gradF(U,λ,v,Y) = sum(Yl * Yl' * U * Diagonal(λ./vl./(λ.+vl)) for (Yl,vl) in zip(Y,v))
F(U,λ,v,Y) = sum(norm(sqrt(Diagonal(λ./vl./(λ.+vl)))*U'*Yl)^2 for (Yl,vl) in zip(Y,v))

updateU_mm(U,λ,v,Y) = polar(gradF(U,λ,v,Y))
updateU_pga(U,λ,v,Y,α) = α == Inf ? polar(gradF(U,λ,v,Y)) : polar(U + α*gradF(U,λ,v,Y))
function updateU_sga(U,λ,v,Y,maxsearches,stepsize,contraction,tol)
    dFdU = gradF(U,λ,v,Y)
    ∇F = dFdU - U*(dFdU'U)
    
    F0, FΔ = F(U,λ,v,Y), tol * norm(∇F)^2
    for m in 1:maxsearches
        Δ = stepsize * contraction^(m-1)
        (F(geodesic(U,∇F,Δ),λ,v,Y) >= F0 + Δ * FΔ) && return geodesic(U,∇F,Δ)
    end
    @debug "Exceeded maximum line search iterations. Accuracy not guaranteed."
    Δ = stepsize * contraction^maxsearches
    return geodesic(U,∇F,Δ)
end
skew(A) = (A'-A)/2
function geodesic(U,X,t)
    k = size(U,2)

    A = skew(U'X)
    Q,R = qr(X - U*(U'X))

    MN = exp(t*[A -R'; R zeros(k,k)])[:,1:k]
    M, N = MN[1:k,:], MN[k+1:end,:]

    return U*M + Matrix(Q)*N
end

# Updates: λ
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
    length(λcritical) == 1 && return mid(interval(only(λcritical)))

    # Return maximizer
    return _argmax(mid.(interval.(λcritical))) do λ
        -sum(n[l]*log(λ+v[l]) + β[l]/(λ+v[l]) for l in 1:L)
    end
end

end
