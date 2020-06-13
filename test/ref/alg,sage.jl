# algorithms: sage

# todo:
# + block version for shared noise variances
# + sometimes root-finding finds no solutions, figure out how to handle

module SAGE

include("polyratio.jl")
include("utils.jl")

using BlockArrays: AbstractBlockArray, eachblock
using BlockArrays: BlockArray
using LinearAlgebra: svd, SVD, Diagonal, /, norm

using Polynomials: Poly, poly
using .PolynomialRatios
using .Utils: posroots

# Updates
updateF(F,v,Y) = updateF(svd(F),v,Y)
updateF(F::SVD,v,Y::BlockArray) = updateF(F,v,Matrix(Y))
function updateF(F::SVD,v,Y)  # todo: use memory more carefully
    U, θ, V = F

    θ2 = θ.^2
    ηt = permutedims(sqrt.(v))
    vt = permutedims(v)

    Ztil = θ ./ ηt ./ (θ2 .+ vt) .* (U'*Y)
    Γsum = Diagonal([sum(1/(θ2j+vi) for vi in v) for θ2j in θ2])

    return ( (Y*(Ztil./ηt)') / (Ztil*Ztil'+Diagonal(Γsum)) ) * V'
end

updatev(F,Y) = updatev(svd(F),Y)
updatev(F::SVD,Y) = [_updatevi(F,yi) for yi in eachcol(Y)]
function _updatevi(F::SVD,yi,tol=1e-14)
    U, θ, _ = F
    d, k = size(F)

    Uyi = U'yi
    Li  = v -> (-(d-k)*log(v) - norm(yi-U*Uyi)^2/v
        - sum(log(θ[j]^2+v) + abs2(Uyi[j])/(θ[j]^2+v) for j in 1:k))
    Lip = (Poly([norm(yi-U*Uyi)^2,-(d-k)]) // poly(zeros(2))
        - sum(Poly([θ[j]^2-abs2(Uyi[j]),1.]) // poly(fill(-θ[j]^2,2)) for j in 1:k))

    criticalpoints = posroots(numerator(Lip), tol)
    optpoint = argmax(Li.(criticalpoints))

    return criticalpoints[optpoint]
end

updatev(F::SVD,Y::AbstractBlockArray) =
    vcat([fill(_updatevl(F,Yl),size(Yl,2)) for Yl in eachblock(Y)]...)
updatev(F::SVD,Y::BlockArray) =
    vcat([fill(_updatevl(F,Yl),size(Yl,2)) for Yl in Y.blocks]...)
function _updatevl(F::SVD,Yl,tol=1e-14)
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

end
