## Types

# Model parameters (to be estimated)
struct HetPPCA{S<:Number,T<:Real}
    U::Matrix{S}   # eigvecs of FF' (factor/spike covariance)
    λ::Vector{T}   # eigvals of FF' (factor/spike covariance)
    Vt::Matrix{S}  # (transposed) eigvecs of F'F (i.e., right singvecs of F)
    v::Vector{T}   # block noise variances
    function HetPPCA{S,T}(U,λ,Vt,v) where {S<:Number,T<:Real}
        size(U,2) == length(λ) || throw(DimensionMismatch("U has dimensions $(size(U)) but λ has length $(length(λ))"))
        size(Vt,1) == length(λ) || throw(DimensionMismatch("Vt has dimensions $(size(Vt)) but λ has length $(length(λ))"))
        new{S,T}(U,λ,Vt,v)
    end
end
HetPPCA(U::Matrix{S},λ::Vector{T},Vt::Matrix{S},v::Vector{T}) where {S<:Number,T<:Real} = HetPPCA{S,T}(U,λ,Vt,v)
function HetPPCA(U::AbstractMatrix,λ::AbstractVector,Vt::AbstractMatrix,v::AbstractVector)
    S = promote_type(eltype(U),eltype(Vt))
    T = promote_type(eltype(λ),eltype(v))
    HetPPCA(convert(Matrix{S},U),
        convert(Vector{T},λ),
        convert(Matrix{S},Vt),
        convert(Vector{T},v))
end
function HetPPCA(F::AbstractMatrix,v::AbstractVector)
    U,s,V = svd(F)
    return HetPPCA(U,s.^2,V',v)
end
Base.:(==)(F::HetPPCA, G::HetPPCA) = all(f -> getfield(F, f) == getfield(G, f), 1:nfields(F))
function Base.getproperty(M::HetPPCA, d::Symbol)
    if d === :F
        U = getfield(M, :U)
        λ = getfield(M, :λ)
        Vt = getfield(M, :Vt)
        return U*sqrt(Diagonal(λ))*Vt
    else
        return getfield(M, d)
    end
end

Base.propertynames(M::HetPPCA, private::Bool=false) =
    private ? (:F, fieldnames(typeof(M))...) : (:F, :U, :λ, :Vt, :v)

# Update methods
struct RootFinding end
struct ExpectationMaximization end
struct MinorizeMaximize end
struct ProjectedGradientAscent{T}
    stepsize::T
end
struct StiefelGradientAscent{T}
    stepsize::T
end
struct DifferenceOfConcave end
struct QuadraticSolvableMinorizer end
struct CubicSolvableMinorizer end
struct QuadraticMinorizer end
struct OptimalQuadraticMinorizer end

# Step sizes
struct InverseLipschitz{T}
    bound::T
end
InverseLipschitz() = InverseLipschitz(LipBoundU2)   # default bound
struct ArmijoSearch{S<:Integer,T<:AbstractFloat}
    maxsearches::S  # maximum number of line searches
    stepsize::T     # initial stepsize
    contraction::T  # contraction factor
    tol::T          # tolerance for sufficient decrease
end