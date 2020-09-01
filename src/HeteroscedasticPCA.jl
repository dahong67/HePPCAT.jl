module HeteroscedasticPCA

using IdentityRanges: IdentityRange
using IntervalArithmetic: interval, mid
using IntervalRootFinding: Newton, roots
using LinearAlgebra: Diagonal, I, opnorm, norm, qr, svd, /
using Logging: @warn

# findmax from https://github.com/cmcaine/julia/blob/argmax-2-arg-harder/base/reduce.jl#L704-L705
# argmax from https://github.com/cmcaine/julia/blob/argmax-2-arg-harder/base/reduce.jl#L830
# part of pull request https://github.com/JuliaLang/julia/pull/35316
_findmax(f, domain) = mapfoldl(x -> (f(x), x), _rf_findmax, domain)
_rf_findmax((fm, m), (fx, x)) = isless(fm, fx) ? (fx, x) : (fm, m)
_argmax(f, domain) = _findmax(f, domain)[2]

# Types
struct HPPCA{S<:Number,T<:Real}
    U::Matrix{S}   # eigvecs of FF' (factor/spike covariance)
    λ::Vector{T}   # eigvals of FF' (factor/spike covariance)
    Vt::Matrix{S}  # (transposed) eigvecs of F'F (i.e., right singvecs of F)
    v::Vector{T}   # block noise variances
    function HPPCA{S,T}(U,λ,Vt,v) where {S<:Number,T<:Real}
        size(U,2) == length(λ) || throw(DimensionMismatch("U has dimensions $(size(U)) but λ has length $(length(λ))"))
        size(Vt,1) == length(λ) || throw(DimensionMismatch("Vt has dimensions $(size(Vt)) but λ has length $(length(λ))"))
        new{S,T}(U,λ,Vt,v)
    end
end
HPPCA(U::Matrix{S},λ::Vector{T},Vt::Matrix{S},v::Vector{T}) where {S<:Number,T<:Real} = HPPCA{S,T}(U,λ,Vt,v)
function HPPCA(U::AbstractMatrix,λ::AbstractVector,Vt::AbstractMatrix,v::AbstractVector)
    S = promote_type(eltype(U),eltype(Vt))
    T = promote_type(eltype(λ),eltype(v))
    HPPCA(convert(Matrix{S},U),
        convert(Vector{T},λ),
        convert(Matrix{S},Vt),
        convert(Vector{T},v))
end
function HPPCA(F::AbstractMatrix,v::AbstractVector)
    U,s,V = svd(F)
    return HPPCA(U,s.^2,V',v)
end
Base.:(==)(F::HPPCA, G::HPPCA) = all(f -> getfield(F, f) == getfield(G, f), 1:nfields(F))

struct RootFinding end
struct ExpectationMaximization end
struct MinorizeMaximize end
struct ProjectedGradientAscent{T}
    stepsize::T
end
struct StiefelGradientAscent{T}
    stepsize::T
end

# Step sizes
struct InverseLipschitz end
struct ArmijoSearch{S<:Integer,T<:AbstractFloat}
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
        updateU!(M,Y,StiefelGradientAscent(ArmijoSearch(max_line,α,β,σ)))
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:λ), getfield.(MM,:v)
end

# log-likelihood (todo: add constant)
function loglikelihood(M,Y)
    d, k = size(M.U)
    n, L = size.(Y,2), length(Y)
    return 1/2*sum(1:L) do l
        norm(sqrt(Diagonal((M.λ./M.v[l])./(M.λ .+ M.v[l])))*M.U'Y[l])^2 -
        n[l]*sum(log.(M.λ .+ M.v[l])) - n[l]*(d-k)*log(M.v[l]) - norm(Y[l])^2/M.v[l]
    end
end

# v updates
function updatev!(M::HPPCA,Y,method)
    for (l,Yl) in enumerate(Y)
        M.v[l] = updatevl(M.v[l],M.U,M.λ,Yl,method)
    end
    return M
end
function updatevl(vl,U,λ,Yl,::RootFinding)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients and check edge case
    α = [j == 0 ? d-k : 1 for j in IdentityRange(0:k)]
    β = [j == 0 ? norm(Yl-U*U'Yl)^2/nl : norm(U[:,j]'Yl)^2/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    if all(iszero(β[j]) for j in 0:k if iszero(γ[j]))
        return zero(vl)
    end

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
function updatevl(vl,U,λ,Yl,::ExpectationMaximization)
    d, k = size(U)
    nl = size(Yl,2)

    return (norm(Yl - U*Diagonal(λ./(λ.+vl))*U'*Yl)^2 + vl*nl*sum(λ./(λ.+vl)))/(d*nl)
end

# F updates
function updateF!(M::HPPCA,Y,::ExpectationMaximization)
    n, L = size.(Y,2), length(Y)
    Λ = Diagonal(M.λ)
    Γ = [inv(Λ + M.v[l]*I) for l in 1:L]
    Z = [Γ[l]*sqrt(Λ)*M.U'*Y[l] for l in 1:L]
    num = sum(Y[l]*Z[l]'/M.v[l] for l in 1:L)
    den = sum(Z[l]*Z[l]'/M.v[l] + n[l]*Γ[l] for l in 1:L)

    F = svd((num / den) * M.Vt)
    M.U .= F.U
    M.λ .= F.S.^2
    M.Vt .= F.Vt
    return M
end

# U updates
function polar(A)
    F = svd(A)
    return F.U*F.Vt
end
gradF(U,λ,v,Y) = sum(Yl * Yl' * U * Diagonal(λ./vl./(λ.+vl)) for (Yl,vl) in zip(Y,v))
F(U,λ,v,Y) = 1/2*sum(norm(sqrt(Diagonal(λ./vl./(λ.+vl)))*U'*Yl)^2 for (Yl,vl) in zip(Y,v))
function LipBoundU1(M::HPPCA,Y)
    L, λmax = length(M.v), maximum(M.λ)
    return sum(norm(Y[l])^2*λmax/M.v[l]/(λmax+M.v[l]) for l in 1:L)
end
function LipBoundU2(M::HPPCA,Y)
    L, λmax = length(M.v), maximum(M.λ)
    return sum(opnorm(Y[l])^2*λmax/M.v[l]/(λmax+M.v[l]) for l in 1:L)
end

updateU!(M::HPPCA,Y,::MinorizeMaximize) = (M.U .= polar(gradF(M.U,M.λ,M.v,Y)); M)
function updateU!(M::HPPCA,Y,pga::ProjectedGradientAscent{<:Number})
    if pga.stepsize == Inf
        M.U .= polar(gradF(M.U,M.λ,M.v,Y))
    else
        M.U .= polar(M.U + pga.stepsize*gradF(M.U,M.λ,M.v,Y))
    end
    return M
end
updateU!(M::HPPCA,Y,pga::ProjectedGradientAscent{InverseLipschitz}) =
    updateU!(M,Y,ProjectedGradientAscent(1/LipBoundU2(M,Y)))
function updateU!(M::HPPCA,Y,sga::StiefelGradientAscent{<:ArmijoSearch})
    params = sga.stepsize
    
    dFdU = gradF(M.U,M.λ,M.v,Y)
    ∇F = dFdU - M.U*(dFdU'M.U)

    F0, FΔ = F(M.U,M.λ,M.v,Y), params.tol * norm(∇F)^2
    for m in 1:params.maxsearches
        Δ = params.stepsize * params.contraction^(m-1)
        if F(geodesic(M.U,∇F,Δ),M.λ,M.v,Y) >= F0 + Δ * FΔ
            M.U .= geodesic(M.U,∇F,Δ)
            return M
        end
    end
    @warn "Exceeded maximum line search iterations. Accuracy not guaranteed."
    Δ = params.stepsize * params.contraction^params.maxsearches
    M.U .= geodesic(M.U,∇F,Δ)
    return M
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
function updateλ!(M::HPPCA,Y,method)
    for (j,uj) in enumerate(eachcol(M.U))
        M.λ[j] = updateλj(M.λ[j],uj,M.v,Y,method)
    end
    return M
end
function updateλj(λj,uj,v,Y,::RootFinding)
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

end
