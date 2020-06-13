module SGD

include("polyratio.jl")

using LinearAlgebra: svd, SVD, Diagonal, /, norm
using LinearAlgebra
using Polynomials: Poly, poly, roots
using .PolynomialRatios

function polar(A)
    U,_,V = svd(A)
    return U*V'
end

∂h(U,θ2,v,Y) = sum(yi * yi' * U * Diagonal([θj2/σi2/(θj2+σi2) for θj2 in θ2]) for (yi,σi2) in zip(Y,v))

_F(U,θ2,σ2,Y) = sum((yi' * U) * Diagonal([θj2/σℓ2/(θj2+σℓ2) for θj2 in θ2]) * (U'*yi) for (yi,σℓ2) in zip(Y,σ2))

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

updatev(U,θ2,Y) = [_updatevi(U,sqrt.(θ2),yi) for yi in Y]
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
    c = [(uj'*yi)^2 for yi in Y]
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

end
