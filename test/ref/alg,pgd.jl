module PGD

include("polyratio.jl")

using LinearAlgebra: svd, SVD, Diagonal, /, norm
using Polynomials: Poly, poly, roots
using .PolynomialRatios

updateL(ynorms,θ2,σ2) = sum(
        ynorm^2*maximum([θj2/σℓ2/(θj2+σℓ2) for θj2 in θ2])
        for (ynorm,σℓ2) in zip(ynorms,σ2)
)

function computeYcolnorms(Y)
    return [norm(Y[i]) for i=1:length(Y)]
end

function polar(A)
    U,_,V = svd(A)
    return U*V'
end

∂h(U,θ2,v,Y) = sum(yi * yi' * U * Diagonal([θj2/σi2/(θj2+σi2) for θj2 in θ2]) for (yi,σi2) in zip(Y,v))

# Updates
updateU(U,θ2,v,Y,α) = polar(α < Inf ? U + α*∂h(U,θ2,v,Y) : ∂h(U,θ2,v,Y))

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
     _fmin(1/sum(size.(Y,2)) * ones(length(c)),c,v)
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
