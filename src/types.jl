## Types

# Model parameters (to be estimated)
"""
    HePPCATModel{S<:Number,T<:Real}

Model parameters for probabilistic PCA with noise that is heteroscedastic across samples.
This is the return type of [`heppcat(_)`](@ref), the corresponding estimation function.
# Properties
+ `F  :: Matrix`    factor matrix (computed via `F = U*sqrt(Diagonal(λ))*Vt`)
+ `U  :: Matrix{S}` eigenvectors of factor covariance `F*F'`
+ `λ  :: Vector{T}` eigenvalues of factor covariance `F*F'` (spike eigenvalues)
+ `Vt :: Matrix{S}` (transposed) eigenvectors of `F'*F` (i.e., right singular vectors of `F`)
+ `v  :: Vector{T}` noise variances
"""
struct HePPCATModel{S<:Number,T<:Real}
    U::Matrix{S}   # eigvecs of FF' (factor/spike covariance)
    λ::Vector{T}   # eigvals of FF' (factor/spike covariance)
    Vt::Matrix{S}  # (transposed) eigvecs of F'F (i.e., right singvecs of F)
    v::Vector{T}   # block noise variances
    function HePPCATModel{S,T}(U,λ,Vt,v) where {S<:Number,T<:Real}
        size(U,2) == length(λ) || throw(DimensionMismatch("U has dimensions $(size(U)) but λ has length $(length(λ))"))
        size(Vt,1) == length(λ) || throw(DimensionMismatch("Vt has dimensions $(size(Vt)) but λ has length $(length(λ))"))
        new{S,T}(U,λ,Vt,v)
    end
end
HePPCATModel(U::Matrix{S},λ::Vector{T},Vt::Matrix{S},v::Vector{T}) where {S<:Number,T<:Real} = HePPCATModel{S,T}(U,λ,Vt,v)

"""
    HePPCATModel(U::AbstractMatrix,λ::AbstractVector,Vt::AbstractMatrix,v::AbstractVector)

Construct HePPCATModel object from factor eigenstructure and noise variances.
"""
function HePPCATModel(U::AbstractMatrix,λ::AbstractVector,Vt::AbstractMatrix,v::AbstractVector)
    S = promote_type(eltype(U),eltype(Vt))
    T = promote_type(eltype(λ),eltype(v))
    HePPCATModel(convert(Matrix{S},U),
        convert(Vector{T},λ),
        convert(Matrix{S},Vt),
        convert(Vector{T},v))
end

"""
    HePPCATModel(F::AbstractMatrix,v::AbstractVector)

Construct HePPCATModel object from factor matrix and noise variances.
"""
function HePPCATModel(F::AbstractMatrix,v::AbstractVector)
    U,s,V = svd(F)
    return HePPCATModel(U,s.^2,V',v)
end

Base.:(==)(F::HePPCATModel, G::HePPCATModel) = all(f -> getfield(F, f) == getfield(G, f), 1:nfields(F))

function Base.getproperty(M::HePPCATModel, d::Symbol)
    if d === :F
        U = getfield(M, :U)
        λ = getfield(M, :λ)
        Vt = getfield(M, :Vt)
        return U*sqrt(Diagonal(λ))*Vt
    else
        return getfield(M, d)
    end
end

Base.propertynames(M::HePPCATModel, private::Bool=false) =
    private ? (:F, fieldnames(typeof(M))...) : (:F, :U, :λ, :Vt, :v)

# Update methods
"""
    RootFinding

Root-finding (global maximization) update method.
No fields.
"""
struct RootFinding end

"""
    ExpectationMaximization

Expectation maximization update method.
No fields.
"""
struct ExpectationMaximization end

"""
    MinorizeMaximize

Minorize maximize update method.
No fields.
"""
struct MinorizeMaximize end

"""
    ProjectedGradientAscent{T}

Projected gradient ascent update method.
One field: `stepsize::T`.
"""
struct ProjectedGradientAscent{T}
    stepsize::T
end

"""
    ProjectedVariance{S,T}

Update variance using `method` then project the variance
so that the variance iterate does not go below `varfloor`.
"""
struct ProjectedVariance{S,T}
    method::S
    varfloor::T
end

"""
    StiefelGradientAscent{T}

Stiefel gradient ascent update method.
One field: `stepsize::T`.
"""
struct StiefelGradientAscent{T}
    stepsize::T
end

"""
    DifferenceOfConcave

Difference of concave update method.
No fields.
"""
struct DifferenceOfConcave end

"""
    QuadraticSolvableMinorizer

Minorize maximize update with quadratic solvable minorizer.
No fields.
"""
struct QuadraticSolvableMinorizer end

"""
    CubicSolvableMinorizer

Minorize maximize update with cubic solvable minorizer.
No fields.
"""
struct CubicSolvableMinorizer end

"""
    QuadraticMinorizer

Minorize maximize update using quadratic minorizer.
No fields.
"""
struct QuadraticMinorizer end

"""
    OptimalQuadraticMinorizer

Minorize maximize update using quadratic minorizer with optimized curvature.
No fields.
"""
struct OptimalQuadraticMinorizer end

# Step sizes
"""
    InverseLipschitz{T}

Inverse Lipschitz step size using bound function specified by field `bound::T`.
Default choice is `bound=LipBoundU2`.
"""
struct InverseLipschitz{T}
    bound::T
end
InverseLipschitz() = InverseLipschitz(LipBoundU2)   # default bound

"""
    ArmijoSearch{S<:Integer,T<:AbstractFloat}

Armijo line search with parameters:
+ `maxsearches :: S`  maximum number of line searches
+ `stepsize    :: T`  initial stepsize
+ `contraction :: T`  contraction factor
+ `tol         :: T`  tolerance for sufficient decrease
"""
struct ArmijoSearch{S<:Integer,T<:AbstractFloat}
    maxsearches::S  # maximum number of line searches
    stepsize::T     # initial stepsize
    contraction::T  # contraction factor
    tol::T          # tolerance for sufficient decrease
end
