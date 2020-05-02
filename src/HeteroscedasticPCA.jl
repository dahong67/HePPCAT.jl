module HeteroscedasticPCA

include("../test/ref/polyratio.jl")
include("../test/ref/utils.jl")

using LinearAlgebra: svd, SVD, Diagonal, /, norm

using Polynomials: Poly, poly
using .PolynomialRatios
using .Utils: posroots

using Polynomials: roots
using LinearAlgebra

using Logging

# Types
struct HPPCA{T<:AbstractFloat}
    U::Matrix{T}
    θ::Vector{T}
    v::Vector{T}
    function HPPCA{T}(U,θ,v) where {T<:AbstractFloat}
        size(U,2) == length(θ) || throw(DimensionMismatch("U has dimensions $(size(U)) but θ has length $(length(θ))"))
        new{T}(U,θ,v)
    end
end
HPPCA(U::Matrix{T},θ::Vector{T},v::Vector{T}) where {T<:AbstractFloat} = HPPCA{T}(U,θ,v)
function HPPCA(F::Matrix{T},v::Vector{T}) where {T<:AbstractFloat}
    U,θ,_ = svd(F)
    return HPPCA(U,θ,v)
end

# PPCA
function ppca(Y,k,iters,init,::Val{:sage})
    M = HPPCA(init,zeros(length(Y)))
    MM = [deepcopy(M)]
    _Vt = svd(init).Vt
    _VVt = [_Vt]
    for t = 1:iters
        updatev!(M,Y,Val(:roots))
        _Vt = updateF!(M,Y,Val(:em),_Vt)
        push!(MM, deepcopy(M))
        push!(_VVt,copy(_Vt))
    end
    return [SVD(M.U,M.θ,_Vt) for (M,_Vt) in zip(MM,_VVt)], getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:mm})
    M = HPPCA(init,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,Val(:oldflatroots))
        updateθ2!(M,Y,Val(:roots))
        updateU!(M,Y,Val(:mm))
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:θ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:pgd})
    Ynorms = vec(mapslices(norm,hcat(Y...),dims=1))
    M = HPPCA(init,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,Val(:oldflatroots))
        updateθ2!(M,Y,Val(:roots))
        L = sum(ynorm*maximum([θj2/σℓ2/(θj2+σℓ2) for θj2 in M.θ])
            for (ynorm,σℓ2) in zip(Ynorms,M.v))
        updateU!(M,Y,Val(:pgd),L)
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:θ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:sgd},max_line=50,α=0.8,β=0.5,σ=1)
    M = HPPCA(init,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,Val(:oldflatroots))
        updateθ2!(M,Y,Val(:roots))
        updateU!(M,Y,Val(:sgd),α,β,σ,max_line,t)
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:θ), getfield.(MM,:v)
end

# Updates: F
function updateF!(M::HPPCA,YY,::Val{:em},Vt)
    U, θ, vv = M.U, M.θ, M.v
    n = size.(YY,2)
    v, Y = vcat(fill.(vv,n)...), hcat(YY...)

    θ2 = θ.^2
    ηt = permutedims(sqrt.(v))
    vt = permutedims(v)

    Ztil = θ ./ ηt ./ (θ2 .+ vt) .* (U'*Y)
    Γsum = Diagonal([sum(1/(θ2j+vi) for vi in v) for θ2j in θ2])

    F = svd( ( (Y*(Ztil./ηt)') / (Ztil*Ztil'+Diagonal(Γsum)) ) * Vt )
    M.U .= F.U
    M.θ .= F.S
    return F.Vt
end

# Updates: v
function updatev!(M::HPPCA,Y,method)
    for (l,Yl) in enumerate(Y)
        M.v[l] = updatevl(M.v[l],M.U,M.θ,Yl,method)
    end
end
function updatevl(vl,U,θ,Yl,::Val{:roots},tol=1e-14)
    d, k = size(U)
    nl = size(Yl,2)

    UYl = U'Yl
    Li  = v -> (-nl*(d-k)*log(v) - norm(Yl-U*UYl)^2/v
        - sum(nl*log(θ[j]^2+v) + sum(abs2,UYl[j,:])/(θ[j]^2+v) for j in 1:k))
    Lip = (Poly([norm(Yl-U*UYl)^2,-nl*(d-k)]) // poly(zeros(2))
        - sum(Poly([nl*θ[j]^2-sum(abs2,UYl[j,:]),nl]) // poly(fill(-θ[j]^2,2)) for j in 1:k))

    criticalpoints = posroots(numerator(Lip), tol)
    optpoint = argmax(Li.(criticalpoints))

    return criticalpoints[optpoint]
end
function updatevl(vi,U,θ2,yi,::Val{:oldflatroots})  # only really for individual samples, not blocks
    θ = sqrt.(θ2)
    d, k = size(U)

    Uyi = U'yi
    Lip = Poly([(norm(yi)^2-norm(Uyi)^2),-(d-k)],:v) // poly(zeros(2),:v) -
        sum(Poly([θ[j]^2-abs2(Uyi[j]),1.],:v) // poly(fill(-θ[j]^2,2),:v) for j in 1:k)

    allroots = roots(numerator(Lip))
    posroots = real.(filter(x -> real(x) ≈ x && real(x) > zero(real(x)),allroots))

    optroot = argmax([
        -(d-k)*log(v) -
        (norm(yi)^2 - norm(Uyi)^2)/v -
        sum(log(θ[j]^2+v) + abs2(Uyi[j])/(θ[j]^2+v) for j in 1:k)
        for v in posroots])

    return posroots[optroot]
end

# Updates: U
polar(A) = polar(svd(A))
polar(A::SVD) = A.U*A.V'
∂h(U,θ2,v,Y) = sum(yi * yi' * U * Diagonal([θj2/σi2/(θj2+σi2) for θj2 in θ2]) for (yi,σi2) in zip(Y,v))

updateU!(M::HPPCA,Y,::Val{:mm}) = (M.U .= polar(∂h(M.U,M.θ,M.v,Y)))
updateU!(M::HPPCA,Y,::Val{:pgd},α) = (M.U .= polar(M.U + α*∂h(M.U,M.θ,M.v,Y)))
function updateU!(M::HPPCA,Y,::Val{:sgd},α,β,σ,max_line,t)
    U, θ2, v = M.U, M.θ, M.v
    d,k = size(U)
    grad = ∂h(U,θ2,v,Y)
    ∇h = grad - U*(grad'U)

    f_U = _F(U,θ2,v,Y)

    η=1
    iter = 0
    while (f_U - _F(R(U,α*∇h,η),θ2,v,Y)) > -σ * tr(∇h'*(η*α*∇h))
        η=β*η
        iter = iter + 1
        if(iter > max_line)
            @warn "Iter $t Exceeded maximum line search iterations. Accuracy not guaranteed."
            break
        end
    end
    return M.U .= R(U,α*∇h,η)
end
_F(U,θ2,σ2,Y) = sum((yi' * U) * Diagonal([θj2/σℓ2/(θj2+σℓ2) for θj2 in θ2]) * (U'*yi) for (yi,σℓ2) in zip(Y,σ2))
function R(U,∇h,η)
    d,k = size(U)
    A = U'∇h
    A = 0.5*(A' - A)
    K = ∇h - U*(U'∇h)
    _Q,R = qr(K); Q = Matrix(_Q)
    MN = exp(η*[A -R'; R zeros(k,k)])[:,1:k]
    M, N = MN[1:k,:], MN[k+1:end,:]
    return U*M + Q*N
end

# Updates: θ
function updateθ2!(M::HPPCA,Y,method)
    for (j,uj) in enumerate(eachcol(M.U))
        M.θ[j] = updateθ2j(M.θ[j],uj,M.v,Y,method)
    end
end
function updateθ2j(θj,uj,σ2,Y,::Val{:roots})
    c = [(uj'*yi)^2 for yi in Y]
    p = 1/sum(size.(Y,2)) * ones(length(c))
    L = length(p)
    fpnum = sum(
        p[ℓ]*poly([
                c[ℓ]-σ2[ℓ],
                (-σ2[ℓp] for ℓp in 1:L if ℓp != ℓ)...,
                (-σ2[ℓp] for ℓp in 1:L if ℓp != ℓ)...
                ])
        for ℓ in 1:L)
    posroots = filter(x -> real(x) ≈ x && real(x) > 0.0,roots(fpnum))
    cand = [0.; real.(posroots)]
    ind = argmin(map(cand) do x
        sum(pℓ*(log(x+σℓ2)+cℓ/(x+σℓ2)) for (pℓ,cℓ,σℓ2) in zip(p,c,σ2))
    end)
    return cand[ind]
end

end
