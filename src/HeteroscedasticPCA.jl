module HeteroscedasticPCA

include("../test/ref/polyratio.jl")
include("../test/ref/utils.jl")

using LinearAlgebra: svd, SVD, Diagonal, /, norm

using Polynomials: Poly, poly
using .PolynomialRatios
using .Utils: posroots

using Polynomials: roots
using LinearAlgebra

# PPCA
function ppca(Y,k,iters,init,::Val{:sage})
    F = Vector{typeof(init)}(undef,iters+1)
    v = Vector{Vector{eltype(init)}}(undef,iters+1)
    F[1] = copy(init)
    for t in 1:iters
        _Ft = svd(F[t])
        vprev = t > 1 ? v[t-1] : zeros(length(Y))  # hack for now until we properly handle init v
        v[t] = updatev(vprev,_Ft,Y,Val(:roots))
        F[t+1] = updateF(_Ft,v[t],Y,Val(:em))
    end
    vprev = iters > 0 ? v[iters] : zeros(length(Y))  # hack for now until we properly handle init v
    v[end] = updatev(vprev,F[end],Y,Val(:roots))

    return F, v
end

# Updates: F
updateF(F,v,Y,method::Val{:em}) = updateF(svd(F),v,Y,method)
function updateF(F::SVD,vv,YY,::Val{:em})  # todo: use memory more carefully
    n = size.(YY,2)
    v, Y = vcat(fill.(vv,n)...), hcat(YY...)
    U, θ, V = F

    θ2 = θ.^2
    ηt = permutedims(sqrt.(v))
    vt = permutedims(v)

    Ztil = θ ./ ηt ./ (θ2 .+ vt) .* (U'*Y)
    Γsum = Diagonal([sum(1/(θ2j+vi) for vi in v) for θ2j in θ2])

    return ( (Y*(Ztil./ηt)') / (Ztil*Ztil'+Diagonal(Γsum)) ) * V'
end

# Updates: v
updatev(v,F,Y,method::Val{:roots}) = updatev(v,svd(F),Y,method)
updatev(v,F::SVD,Y,method::Val{:roots}) = [updatevl(vl,F,Yl,method) for (vl,Yl) in zip(v,Y)]
function updatevl(vl,F::SVD,Yl,::Val{:roots},tol=1e-14)
    U, θ, _ = F
    d, k = size(F)
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

# MM
function ppca(Y,k,iters,init,::Val{:mm})
    U = Vector{typeof(init)}(undef,iters+1)
    θ2 = Vector{Vector{eltype(init)}}(undef,iters+1)
    v = Vector{Vector{eltype(init)}}(undef,iters+1)
    Q,S,_ = svd(init)
    U[1] = Q[:,1:k]
    θ2[1] = S[1:k]
    for t in 1:iters
        v[t] = updatev(U[t],θ2[t],Y)
        θ2[t+1] = updateθ2(U[t],v[t],Y)
        U[t+1] = updateU(U[t],θ2[t+1],v[t],Y)
    end
    v[end] = updatev(U[end],θ2[end],Y)
    Fhat = U[end] * Diagonal(sqrt.(θ2[end]))
    return Fhat, v
end

function polar(A)
    U,_,V = svd(A)
    return U*V'
end

∂h(U,θ2,v,Y) = sum(yi * yi' * U * Diagonal([θj2/σi2/(θj2+σi2) for θj2 in θ2]) for (yi,σi2) in zip(eachcol(Y),v))

updateU(U,θ2,v,Y) = polar(∂h(U,θ2,v,Y))

updatev(U,θ2,Y) = [_updatevi(U,sqrt.(θ2),yi) for yi in eachcol(Y)]
function _updatevi(U,θ,yi)
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

updateθ2(U,v,Y) = [_updateθ2l(uj,v,Y) for uj in eachcol(U)]
function _updateθ2l(uj,v,Y)
    c = [(uj'*yi)^2 for yi in eachcol(Y)]
     _fmin(1/size(Y,2) * ones(length(c)),c,v)
 end

_f(x;p,c,σ2) = sum(pℓ*(log(x+σℓ2)+cℓ/(x+σℓ2)) for (pℓ,cℓ,σℓ2) in zip(p,c,σ2))

function _fmin(p,c,σ2)
    cand = [0.; _fproots(p,c,σ2)]
    ind = argmin([_f(x,p=p,c=c,σ2=σ2) for x in cand])
    return cand[ind]
end

function _fproots(p,c,σ2)
    L = length(p)

    fpnum = sum(
        p[ℓ]*poly([
                c[ℓ]-σ2[ℓ],
                (-σ2[ℓp] for ℓp in 1:L if ℓp != ℓ)...,
                (-σ2[ℓp] for ℓp in 1:L if ℓp != ℓ)...
                ])
        for ℓ in 1:L)

    posroots = filter(x -> real(x) ≈ x && real(x) > 0.0,roots(fpnum))
    return real.(posroots)
end

# PGD
function ppca(Y,k,iters,init,::Val{:pgd})
    U = Vector{typeof(init)}(undef,iters+1)
    θ2 = Vector{Vector{eltype(init)}}(undef,iters+1)
    v = Vector{Vector{eltype(init)}}(undef,iters+1)
    Q,S,_ = svd(init)
    U[1] = Q[:,1:k]
    θ2[1] = S[1:k]
    Ynorms = computeYcolnorms(Y)
    for t in 1:iters
        v[t] = updatev(U[t],θ2[t],Y)
        θ2[t+1] = updateθ2(U[t],v[t],Y)
        L = updateL(Ynorms,θ2[t+1],v[t])
        U[t+1] = updateU(U[t],θ2[t+1],v[t],Y,L)
    end
    v[end] = updatev(U[end],θ2[end],Y)
    Fhat = U[end] * Diagonal(sqrt.(θ2[end]))
    return Fhat, v
end

updateL(ynorms,θ2,σ2) = sum(
        ynorm*maximum([θj2/σℓ2/(θj2+σℓ2) for θj2 in θ2])
        for (ynorm,σℓ2) in zip(ynorms,σ2)
)

function computeYcolnorms(Y)
    return [norm(Y[:,i]) for i=1:size(Y,2)]
end

updateU(U,θ2,v,Y,α) = polar(U + α*∂h(U,θ2,v,Y))

# SGD
function ppca(Y,k,iters,init,::Val{:sgd},max_line=50,α=0.8,β=0.5,σ=1)
    U = Vector{typeof(init)}(undef,iters+1)
    θ2 = Vector{Vector{eltype(init)}}(undef,iters+1)
    v = Vector{Vector{eltype(init)}}(undef,iters+1)
    Q,S,_ = svd(init)
    U[1] = Q[:,1:k]
    θ2[1] = S[1:k]

    for t in 1:iters
        v[t] = updatev(U[t],θ2[t],Y)
        θ2[t+1] = updateθ2(U[t],v[t],Y)
        U[t+1] = updateU(U[t],θ2[t+1],v[t],Y,α,β,σ,max_line,t)
    end
    v[end] = updatev(U[end],θ2[end],Y)
    Fhat = U[end] * Diagonal(sqrt.(θ2[end]))
    return Fhat, v
end

_F(U,θ2,σ2,Y) = sum((yi' * U) * Diagonal([θj2/σℓ2/(θj2+σℓ2) for θj2 in θ2]) * (U'*yi) for (yi,σℓ2) in zip(eachcol(Y),σ2))

function updateU(U,θ2,v,Y,α,β,σ,max_line,t)
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
            println("Iter $t Exceeded maximum line search iterations. Accuracy not guaranteed.")
            break
        end
    end
    return R(U,α*∇h,η)
end

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

end
