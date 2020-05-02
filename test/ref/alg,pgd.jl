# algorithms: sage

# todo:
# + block version for shared noise variances
# + sometimes root-finding finds no solutions, figure out how to handle

module PGD

include("polyratio.jl")

using LinearAlgebra: svd, SVD, Diagonal, /, norm
using Polynomials: Poly, poly, roots
using .PolynomialRatios
using BlockArrays: AbstractBlockArray, eachblock

# Algorithm
function ppca(Y,k,iters,init)
    U = Vector{typeof(init)}(undef,iters+1)
    θ2 = Vector{Vector{eltype(init)}}(undef,iters+1)
    v = Vector{Vector{eltype(init)}}(undef,iters+1)
    Q,S,_ = svd(init)
    U[1] = Q[:,1:k]
    θ2[1] = S[1:k].^2
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

function polar(A)
    U,_,V = svd(A)
    return U*V'
end

∂h(U,θ2,v,Y) = sum(yi * yi' * U * Diagonal([θj2/σi2/(θj2+σi2) for θj2 in θ2]) for (yi,σi2) in zip(eachcol(Y),v))

# Updates
updateU(U,θ2,v,Y,α) = polar(U + α*∂h(U,θ2,v,Y))
# function updateU(U,θ2,σ2,YY::AbstractBlockArray)  # todo: use memory more carefully
#     nl = [size(Yl,2) for Yl in eachblock(Y)]
#     Λ = [Yl*Yl'/size(Yl,2) for Yl in eachblock(Y)]
#     p = nl./sum(nl)
#     return polar(
#         sum(
#             pℓ*Λℓ*U*Diagonal([θj2/σℓ2/(θj2+σℓ2) for θj2 in θ2])
#             for (pℓ,Λℓ,σℓ2) in zip(p,Λ,σ2)
#         )
# end

updatev(U,θ2,Y) = [_updatevi(U,sqrt.(θ2),yi) for yi in eachcol(Y)]
function _updatevi(U,θ,yi)

    d, k = size(U)

    Uyi = U'yi
    # Lip = -(d-k) // poly(zeros(1),:v) +
    #     (norm(yi)^2-norm(Uyi)^2) // poly(zeros(2),:v) -
    #     sum(1. // poly(fill(-θ[j]^2,1),:v) -
    #         abs2(Uyi[j]) // poly(fill(-θ[j]^2,2),:v) for j in 1:k)
    # currently, PolynomialRatios does not reduce to avoid losing roots
    # so to help out we combine some terms in advance
    # briefly tried BigFloats to see if roots wouldn't be lost but it didn't
    # seem to resolve it. a cause appears to be norm(yi)^2-norm(Uyi)^2 ≈ 0
    # leading to degree going down - todo: document
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

# updatev(F::SVD,Y::AbstractBlockArray) =
#     vcat([fill(_updatevl(F,Yl),size(Yl,2)) for Yl in eachblock(Y)]...)
# function _updatevl(F::SVD,Yl)
#     U, θ, _ = F
#     d, k = size(F)
#     nl = size(Yl,2)
#
#     UYl = U'Yl
#     Lip = Poly([(norm(Yl)^2-norm(UYl)^2),-nl*(d-k)],:v) // poly(zeros(2),:v) -
#         sum(Poly([nl*θ[j]^2-sum(abs2,UYl[j,:]),nl],:v) // poly(fill(-θ[j]^2,2),:v) for j in 1:k)
#
#     allroots = roots(numerator(Lip))
#     posroots = real.(filter(x -> real(x) ≈ x && real(x) > zero(real(x)),allroots))
#
#     optroot = argmax([
#         -nl*(d-k)*log(v) -
#         (norm(Yl)^2 - norm(UYl)^2)/v -
#         sum(nl*log(θ[j]^2+v) + sum(abs2,UYl[j,:])/(θ[j]^2+v) for j in 1:k)
#         for v in posroots])
#
#     return posroots[optroot]
# end
#
# end


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

end
