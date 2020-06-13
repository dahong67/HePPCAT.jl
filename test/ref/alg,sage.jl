module SAGE

include("polyratio.jl")
include("utils.jl")

using LinearAlgebra: svd, SVD, Diagonal, /, norm
using Polynomials: Poly, poly
using .PolynomialRatios
using .Utils: posroots

# Updates
function updateF(F,vblocks,Yblocks)  # todo: use memory more carefully
    Y = hcat(Yblocks...)
    v = vcat(fill.(vblocks,size.(Yblocks,2))...)
    U, θ, V = svd(F)

    θ2 = θ.^2
    ηt = permutedims(sqrt.(v))
    vt = permutedims(v)

    Ztil = θ ./ ηt ./ (θ2 .+ vt) .* (U'*Y)
    Γsum = Diagonal([sum(1/(θ2j+vi) for vi in v) for θ2j in θ2])

    return ( (Y*(Ztil./ηt)') / (Ztil*Ztil'+Diagonal(Γsum)) ) * V'
end

updatev(F,Y) = [_updatevl(F,Yl) for Yl in Y]
function _updatevl(F,Yl,tol=1e-14)
    U, θ, _ = svd(F)
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
