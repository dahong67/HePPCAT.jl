module HeteroscedasticPCA

using LinearAlgebra: Diagonal, norm, qr, svd, tr, /
using Logging

include("../test/ref/polyratio.jl")
include("../test/ref/utils.jl")
using Polynomials: Poly, poly
using .PolynomialRatios
using .Utils: posroots
using Polynomials: roots

# Types
struct HPPCA{T<:AbstractFloat}
    U::Matrix{T}   # eigvecs of FF'
    λ::Vector{T}   # eigvals of FF'
    v::Vector{T}   # block noise variances
    function HPPCA{T}(U,λ,v) where {T<:AbstractFloat}
        size(U,2) == length(λ) || throw(DimensionMismatch("U has dimensions $(size(U)) but λ has length $(length(λ))"))
        new{T}(U,λ,v)
    end
end
HPPCA(U::Matrix{T},λ::Vector{T},v::Vector{T}) where {T<:AbstractFloat} = HPPCA{T}(U,λ,v)

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
    M = HPPCA(svd(init).U,svd(init).S.^2,zeros(length(Y)))
    MM = [deepcopy(M)]
    _Vt = svd(init).Vt
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        _Vt = updateF!(M,Y,ExpectationMaximization(),_Vt)
        push!(MM, deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM, :λ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:mm})
    M = HPPCA(svd(init).U,svd(init).S.^2,zeros(length(Y)))
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
    M = HPPCA(svd(init).U,svd(init).S.^2,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        updateλ!(M,Y,RootFinding())
        L = sum(ynorm*maximum([λj/vi/(λj+vi) for λj in M.λ])   # todo: should ynorm be squared?
            for (ynorm,vi) in zip(Ynorms,M.v))
        updateU!(M,Y,ProjectedGradientAscent(L))
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:λ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:sgd},max_line=50,α=0.8,β=0.5,σ=1.0)
    M = HPPCA(svd(init).U,svd(init).S.^2,zeros(length(Y)))
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
function updateF!(M::HPPCA,YY,::ExpectationMaximization,Vt)
    U, λ, vv = M.U, M.λ, M.v
    n = size.(YY,2)
    v, Y = vcat(fill.(vv,n)...), hcat(YY...)

    ηt = permutedims(sqrt.(v))
    vt = permutedims(v)

    Ztil = sqrt.(λ) ./ ηt ./ (λ .+ vt) .* (U'*Y)
    Γsum = Diagonal([sum(1/(λj+vi) for vi in v) for λj in λ])

    F = svd( ( (Y*(Ztil./ηt)') / (Ztil*Ztil'+Diagonal(Γsum)) ) * Vt )
    M.U .= F.U
    M.λ .= F.S.^2
    return F.Vt
end

# Updates: v
function updatev!(M::HPPCA,Y,method)
    for (l,Yl) in enumerate(Y)
        M.v[l] = updatevl(M.v[l],M.U,M.λ,Yl,method)
    end
end
function updatevl(vl,U,λ,Yl,::RootFinding,tol=1e-14)
    d, k = size(U)
    nl = size(Yl,2)

    UYl = U'Yl
    Li  = v -> (-nl*(d-k)*log(v) - norm(Yl-U*UYl)^2/v
        - sum(nl*log(λ[j]+v) + sum(abs2,UYl[j,:])/(λ[j]+v) for j in 1:k))
    Lip = (Poly([norm(Yl-U*UYl)^2,-nl*(d-k)]) // poly(zeros(2))
        - sum(Poly([nl*λ[j]-sum(abs2,UYl[j,:]),nl]) // poly(fill(-λ[j],2)) for j in 1:k))

    criticalpoints = posroots(numerator(Lip), tol)
    optpoint = argmax(Li.(criticalpoints))

    return criticalpoints[optpoint]
end

# Updates: U
function polar(A)
    F = svd(A)
    return F.U*F.Vt
end
gradF(U,λ,v,Y) = sum(yi * yi' * U * Diagonal(λ./vi./(λ.+vi)) for (yi,vi) in zip(Y,v))
F(U,λ,v,Y) = sum((yi' * U) * Diagonal([λj/vi/(λj+vi) for λj in λ]) * (U'*yi) for (yi,vi) in zip(Y,v))

updateU!(M::HPPCA,Y,::MinorizeMaximize) = (M.U .= polar(gradF(M.U,M.λ,M.v,Y)))
updateU!(M::HPPCA,Y,pga::ProjectedGradientAscent) = (M.U .= polar(M.U + pga.stepsize*gradF(M.U,M.λ,M.v,Y)))
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
    c = [(uj'*yi)^2 for yi in Y]
    p = 1/sum(size.(Y,2)) * ones(length(c))
    L = length(p)
    fpnum = sum(
        p[l]*poly([
                c[l]-v[l],
                (-v[lp] for lp in 1:L if lp != l)...,
                (-v[lp] for lp in 1:L if lp != l)...
                ])
        for l in 1:L)
    posroots = filter(x -> real(x) ≈ x && real(x) > 0.0,roots(fpnum))
    cand = [0.; real.(posroots)]
    ind = argmin(map(cand) do x
        sum(pl*(log(x+vl)+cl/(x+vl)) for (pl,cl,vl) in zip(p,c,v))
    end)
    return cand[ind]
end

end
