module HeteroscedasticPCA

using IntervalArithmetic: interval, mid
using IntervalRootFinding: Newton, roots
using LinearAlgebra: Diagonal, norm, qr, svd, tr, /
using Logging

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
function updateF!(M::HPPCA,YY,::ExpectationMaximization)
    U, λ, Vt, vv = M.U, M.λ, M.Vt, M.v
    n = size.(YY,2)
    v, Y = vcat(fill.(vv,n)...), hcat(YY...)

    ηt = permutedims(sqrt.(v))
    vt = permutedims(v)

    Ztil = sqrt.(λ) ./ ηt ./ (λ .+ vt) .* (U'*Y)
    Γsum = Diagonal([sum(1/(λj+vi) for vi in v) for λj in λ])

    F = svd( ( (Y*(Ztil./ηt)') / (Ztil*Ztil'+Diagonal(Γsum)) ) * Vt )
    M.U .= F.U
    M.λ .= F.S.^2
    M.Vt .= F.Vt
end

# Updates: v
function updatev!(M::HPPCA,Y,method)
    for (l,Yl) in enumerate(Y)
        M.v[l] = updatevl(M.v[l],M.U,M.λ,Yl,method)
    end
end
function updatevl(vl,U,λ,Yl,::RootFinding)
    d, k = size(U)
    nl = size(Yl,2)

    α0, β0 = d-k, norm(Yl-U*U'Yl)^2/nl
    β = [norm(uj'Yl)^2/nl for uj in eachcol(U)]

    vl0opt = β0/α0
    vljopt = max.(zero.(β), β .- λ)
    vmin, vmax = min(vl0opt,vljopt...), max(vl0opt,vljopt...)
    vrange = interval(vmin,vmax)

    tol = 1e-8  # todo: choose tolerance adaptively
    vmax-vmin < tol && return (vmax+vmin)/2
    vcritical = mid.(interval.(roots(v -> β0/v^2-α0/v + sum(β[j]/(λ[j]+v)^2 - 1/(λ[j]+v) for j in 1:k),vrange,Newton,tol)))
    isempty(vcritical) && return vmin
    length(vcritical) == 1 && return only(vcritical)
    Lcritical = map(v -> -(α0*log(v)+β0/v + sum(log(λ[j]+v) + β[j]/(λ[j]+v) for j in 1:k)),vcritical)

    return vcritical[argmax(Lcritical)]
end

# Updates: U
function polar(A)
    F = svd(A)
    return F.U*F.Vt
end
gradF(U,λ,v,Y) = sum(Yl * Yl' * U * Diagonal(λ./vl./(λ.+vl)) for (Yl,vl) in zip(Y,v))
F(U,λ,v,Y) = sum(norm(sqrt(Diagonal(λ./vl./(λ.+vl)))*U'*Yl)^2 for (Yl,vl) in zip(Y,v))

updateU!(M::HPPCA,Y,::MinorizeMaximize) = (M.U .= polar(gradF(M.U,M.λ,M.v,Y)))
function updateU!(M::HPPCA,Y,pga::ProjectedGradientAscent)
    if pga.stepsize < Inf
        M.U .= polar(M.U + pga.stepsize*gradF(M.U,M.λ,M.v,Y))
    else
        M.U .= polar(gradF(M.U,M.λ,M.v,Y))
    end
end
function updateU!(M::HPPCA,Y,sga::StiefelGradientAscent)
    α, β, σ, maxsearches = sga.stepsize, sga.contraction, sga.tol, sga.maxsearches
    U, λ, v = M.U, M.λ, M.v

    dFdU = gradF(U,λ,v,Y)
    ∇F = dFdU - U*(dFdU'U)

    Δ, m = α, 0
    while F(U,λ,v,Y) - F(geodesic(U,∇F,Δ),λ,v,Y) > -σ * Δ * tr(∇F'*∇F)
        Δ *= β
        m += 1
        if m > maxsearches
            @warn "Exceeded maximum line search iterations. Accuracy not guaranteed."
            break
        end
    end
    return M.U .= geodesic(U,∇F,Δ)
end
function geodesic(U,X,t)
    k = size(U,2)

    A = U'X
    A = (A' - A)/2

    Q,R = qr(X - U*(U'X))
    MN = exp(t*[A -R'; R zeros(k,k)])[:,1:k]
    M, N = MN[1:k,:], MN[k+1:end,:]

    return U*M + Matrix(Q)*N
end

# Updates: λ
function updateλ!(M::HPPCA,Y,method)
    for (j,uj) in enumerate(eachcol(M.U))
        M.λ[j] = updateλj(M.λ[j],uj,M.v,Y,method)
    end
end
function updateλj(λj,uj,v,Y,::RootFinding)
    n = size.(Y,2)
    L = length(n)
    β = [norm(uj'Yl)^2 for Yl in Y]

    λjlopt = max.(zero.(β), β .- v)
    λmin,λmax = extrema(λjlopt)
    λrange = interval(λmin,λmax)

    tol = 1e-8  # todo: choose tolerance adaptively
    λmax-λmin < tol && return (λmax+λmin)/2
    λcritical = mid.(interval.(roots(λ -> sum(β[l]/(λ+v[l])^2 - n[l]/(λ+v[l]) for l in 1:L),λrange,Newton,tol)))
    isempty(λcritical) && return λmin
    length(λcritical) == 1 && return only(λcritical)
    Lcritical = map(λ -> -sum(n[l]*log(λ+v[l]) + β[l]/(λ+v[l]) for l in 1:L),λcritical)

    return λcritical[argmax(Lcritical)]
end

end
