module HeteroscedasticPCA

include("../test/ref/polyratio.jl")
include("../test/ref/utils.jl")

using LinearAlgebra: svd, SVD, Diagonal, /, norm

using Polynomials: Poly, poly
using .PolynomialRatios
using .Utils: posroots

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

end
