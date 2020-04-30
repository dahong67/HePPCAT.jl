# utility functions: conventional methods, metrics, data generation

module Utils

using LinearAlgebra
using Random
using Distributions
using Polynomials: Poly, polyder, degree
using IntervalArithmetic: @interval, mid, interval
using IntervalRootFinding: roots, Newton
using Manifolds: Stiefel, uniform_distribution

export UnifSt, Latent, posroots

## Positive root-finding
function posroots(p::Poly{T}, tol::T) where {T}
    bound = _hongbound(p)
    bound < tol && return [tol/2]::Vector{T}

    _roots = roots(p, polyder(p), @interval(zero(bound),bound), Newton, tol)
    return [mid(interval(r)) for r in _roots]::Vector{T}
end

# doi: 10.1006/jsco.1997.0189
function _hongbound(f::Poly{T}) where {T}
    d = degree(f)
    a = sign(f[d]) * f    # make leading coefficient positive
    return 2*maximum(filter(q -> a[q] < zero(a[q]), 0:d)) do q
        minimum(filter(p -> a[p] > zero(a[p]), q+1:d)) do p
            abs(a[q]/a[p])^(1/(p-q))
        end
    end::T
end

## Conventional methods: homoscedastic ppca, weighted pca

# Homoscedastic PPCA
_pos(x) = max(zero(x),x)
function ppca(Y,k)
    d,n = size(Y)
    λ,U = eigen(Hermitian(Y*Y'/n))
    σ2 = sum(λ[1:d-k])/(d-k)
    return U[:,(d-k+1):d]*Diagonal(sqrt.(_pos.(λ[(d-k+1):d] .- σ2)))
end

# todo: remove use of blocks, then uncomment
# # Weighted PCA
# function wpca(Y,k,w2,n)
#     d, L = size(Y,1), length(n)
#
#     # Compute weighted sample covariance
#     YB = BlockArray(Y,[d],n)
#     wΛ = sum(w2[l]*(YB[Block(1,l)]*YB[Block(1,l)]') for l in 1:L) / sum(n)
#
#     # Compute principal eigendecomposition
#     λ,U = eigen(Hermitian(wΛ),(d-k+1):d)
#     return Factor(U,λ)
# end
# w(θ2,η2) = (θ2/η2)/(θ2+η2)

## Metrics: log-likelihood, alignment

# todo: remove use of blocks, then uncomment
#
# using LinearAlgebra, BlockArrays
#
# function ℒ(F,Y,σ2,n)
#     U, θ2 = F
#     d, k = size(F)
#     L = length(n)
#     YB = BlockArray(Y,[d],n)
#
#     term1 = -1/2*sum(norm(YB[Block(1,l)])^2/σ2[l] + n[l]*(d-k)*log(σ2[l]) for l in 1:L)
#     term2 = 1/2*sum(norm((U*Diagonal(sqrt.(w.(θ2,σ2[l]))))'*YB[Block(1,l)])^2 for l in 1:L)
#     term3 = -1/2*sum(n[l]*sum(log(θ2[j]+σ2[l]) for j in 1:k) for l in 1:L)
#     return term1 + term2 + term3
# end
#
# sqdot(F,Fh) = (F.U'Fh.U).^2

## Data generation

# Uniform Stiefel (now from Manifolds.jl)
UnifSt(k,m) = uniform_distribution(Stiefel(m,k),zeros(m,k))

# Latent model
struct Latent{T, R<:AbstractMatrix{T}, S<:AbstractVector{T}} <: Sampleable{Matrixvariate, Continuous}
    F::R    # factor matrix
    η::S    # vector of noise std devs
    Latent{T,R,S}(F::R,η::S) where {T, R<:AbstractMatrix{T}, S<:AbstractVector{T}} =
        any(x -> x < zero(x),η) ? error("η has negative entries") : new(F,η)
end
Latent(F::R,η::S) where {T, R<:AbstractMatrix{T}, S<:AbstractVector{T}} = Latent{T,R,S}(F,η)
function Latent(F::AbstractMatrix{FT},η::AbstractVector{ηT}) where {FT, ηT}
    T = promote_type(FT, ηT)
    Latent(convert(AbstractMatrix{T},F),convert(AbstractVector{T},η))
end

Base.size(d::Latent) = (size(d.F,1),length(d.η))
Base.size(d::Latent, i) = i::Integer <= 2 ? size(d)[i] : 1
Base.eltype(::Type{<:Latent{T}}) where {T} = T
function Distributions._rand!(rng::AbstractRNG, dist::Latent{T}, A::AbstractMatrix{T}) where {T}
    randn!(rng,A)                                   # generate noise
    A .*= permutedims(dist.η)                       # scale by noise std dev

    n, k = size(dist,2), size(dist.F,2)
    mul!(A,dist.F,randn(rng,T,k,n),one(T),one(T))   # add signal
end

end
