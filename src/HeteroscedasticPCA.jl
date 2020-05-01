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

# PPCA
function ppca(Y,k,iters,init,::Val{:sage})
    v, F = [zeros(length(Y))], [svd(init)]
    for t = 1:iters
        push!(v, updatev(v[end],F[end],Y,Val(:roots)))
        push!(F, svd(updateF(F[end],v[end],Y,Val(:em))))
    end
    return F, v
end
function ppca(YY,k,iters,init,::Val{:mm})
    IVt = Matrix{eltype(init)}(I,k,k)
    Q, S, _ = svd(init)
    v, U, θ2 = [zeros(length(YY))], [Q], [S]
    for t = 1:iters
        push!(v, updatev(v[end],SVD(U[end],sqrt.(θ2[end]),IVt),YY,Val(:oldflatroots)))
        push!(θ2, updateθ2(U[end],v[end],YY,Val(:roots)))
        push!(U, updateU(U[end],θ2[end],v[end],YY,Val(:mm)))
    end
    return U, θ2, v
end
function ppca(YY,k,iters,init,::Val{:pgd})
    IVt = Matrix{eltype(init)}(I,k,k)
    Q, S, _ = svd(init)
    v, U, θ2 = [zeros(length(YY))], [Q], [S]
    Ynorms = vec(mapslices(norm,hcat(YY...),dims=1))
    for t in 1:iters
        push!(v, updatev(v[end],SVD(U[end],sqrt.(θ2[end]),IVt),YY,Val(:oldflatroots)))
        push!(θ2, updateθ2(U[end],v[end],YY,Val(:roots)))
        L = sum(ynorm*maximum([θj2/σℓ2/(θj2+σℓ2) for θj2 in θ2[end]])
            for (ynorm,σℓ2) in zip(Ynorms,v[end]))
        push!(U, updateU(U[end],θ2[end],v[end],YY,Val(:pgd),L))
    end
    return U, θ2, v
end
function ppca(YY,k,iters,init,::Val{:sgd},max_line=50,α=0.8,β=0.5,σ=1)
    IVt = Matrix{eltype(init)}(I,k,k)
    Q, S, _ = svd(init)
    v, U, θ2 = [zeros(length(YY))], [Q], [S]
    for t in 1:iters
        push!(v, updatev(v[end],SVD(U[end],sqrt.(θ2[end]),IVt),YY,Val(:oldflatroots)))
        push!(θ2, updateθ2(U[end],v[end],YY,Val(:roots)))
        push!(U, updateU(U[end],θ2[end],v[end],YY,Val(:sgd),α,β,σ,max_line,t))
    end
    return U, θ2, v
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

updatev(v,F::SVD,Y,method::Val{:oldflatroots}) = [updatevi(vi,F,yi,method) for (vi,yi) in zip(v,Y)]
function updatevi(vi,F::SVD,yi,::Val{:oldflatroots})
    U,θ,_ = F
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

updateU(U,θ2,v,Y,::Val{:mm}) = polar(∂h(U,θ2,v,Y))
updateU(U,θ2,v,Y,::Val{:pgd},α) = polar(U + α*∂h(U,θ2,v,Y))
function updateU(U,θ2,v,Y,::Val{:sgd},α,β,σ,max_line,t)
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
    return R(U,α*∇h,η)
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
updateθ2(U,v,Y,method::Val{:roots}) = [updateθ2l(uj,v,Y,method) for uj in eachcol(U)]
function updateθ2l(uj,σ2,Y,::Val{:roots})
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
