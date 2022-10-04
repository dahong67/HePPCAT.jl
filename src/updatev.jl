## v updates

"""
    updatev!(M::HePPCATModel,Y,method)

Update the noise variances `M.v` with respect to data `Y` using `method`.
Internally calls [`updatevl(_)`](@ref) to update each entry.
"""
function updatev!(M::HePPCATModel,Y,method)
    for (l,Yl) in enumerate(Y)
        M.v[l] = updatevl(M.v[l],M.U,M.λ,Yl,method)
    end
    return M
end

"""
    updatevl(vl,U,λ,Yl,method)

Compute update for `l`th noise variance `vl` with respect to data `Yl` using `method`.
"""
function updatevl end

# Update method: Global maximization via root-finding
"""
    updatevl(vl,U,λ,Yl,::RootFinding)

Root-finding (global maximization) update of `vl`.
"""
function updatevl(vl,U,λ,Yl,::RootFinding)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients and check edge case
    UYl = [sum(abs2,Yl'U[:,j]) for j in 1:k]
    UYl0 = nonnegative(sum(abs2,Yl)-sum(UYl))
    α = [j == 0 ? d-k : 1 for j in IdentityRange(0:k)]
    β = [j == 0 ? UYl0/nl : UYl[j]/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    all(iszero(β[j]) for j in 0:k if iszero(γ[j])) && return zero(vl)

    # Find nonnegative critical points
    tol = 1e-8  # todo: choose tolerance adaptively
    vmin, vmax = extrema(β[j]/α[j]-γ[j] for j in 0:k if !(iszero(α[j]) && iszero(β[j])))
    vcritical = roots(interval(vmin,vmax) ∩ interval(zero(vl),Inf),Newton,tol) do v
        - sum(α[j]/(γ[j]+v) for j in 0:k if !iszero(α[j])) + sum(β[j]/(γ[j]+v)^2 for j in 0:k if !iszero(β[j]))
    end

    # Return maximizer
    return _argmax(mid.(interval.(vcritical))) do v
        -(sum(α[j]*log(γ[j]+v) for j in 0:k if !iszero(α[j])) + sum(β[j]/(γ[j]+v) for j in 0:k if !iszero(β[j])))
    end
end

# Update method: Expectation Maximization
"""
    updatevl(vl,U,λ,Yl,::ExpectationMaximization)

Expectation maximization update of `vl`.
"""
function updatevl(vl,U,λ,Yl,::ExpectationMaximization)
    d, k = size(U)
    nl = size(Yl,2)

    UYl = [sum(abs2,Yl'U[:,j]) for j in 1:k]
    UYl0 = nonnegative(sum(abs2,Yl)-sum(UYl))
    β = [j == 0 ? UYl0/nl : UYl[j]/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    
    ρ = sum((oneunit(eltype(λ)).-γ./(γ.+vl)).^2 .* β) + vl*sum(λ./(λ.+vl))
    return ρ/d
end
# Update Method: Projected Variance (any method)
"""
Ensures minimum variance does not go below varfloor for any given method
"""
function updatevl(vl,U,λ,Yl,method::ProjectedVariance)
    return max(updatevl(vl,U,λ,Yl,method.method),method.varfloor)
end

# Update method: Difference of concave approach
"""
    updatevl(vl,U,λ,Yl,::DifferenceOfConcave)

Difference of concave update of `vl`.
"""
function updatevl(vl,U,λ,Yl,::DifferenceOfConcave)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients and check edge case
    UYl = [sum(abs2,Yl'U[:,j]) for j in 1:k]
    UYl0 = nonnegative(sum(abs2,Yl)-sum(UYl))
    α = [j == 0 ? d-k : 1 for j in IdentityRange(0:k)]
    β = [j == 0 ? UYl0/nl : UYl[j]/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    affslope = -sum(α[j]/(γ[j]+vl) for j in 0:k if !iszero(α[j]))
    Ltp = v -> affslope + sum(β[j]/(γ[j]+v)^2 for j in 0:k if !iszero(β[j]))
    if affslope == -Inf || Ltp(zero(vl)) <= zero(vl)
        return zero(vl)
    end

    # Return nonnegative critical point
    tol = 1e-8  # todo: choose tolerance adaptively
    vmax = maximum(sqrt(β[j]/α[j]*(γ[j]+vl)) - γ[j] for j in 0:k if !(iszero(α[j]) && iszero(β[j])))
    return find_zero(Ltp, (zero(vmax),vmax))
end

# Update method: Quadratic solvable minorizer
"""
    updatevl(vl,U,λ,Yl,::QuadraticSolvableMinorizer)

Minorize maximize update of `vl` with quadratic solvable minorizer.
"""
function updatevl(vl,U,λ,Yl,::QuadraticSolvableMinorizer)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients
    UYl = [sum(abs2,Yl'U[:,j]) for j in 1:k]
    UYl0 = nonnegative(sum(abs2,Yl)-sum(UYl))
    α = [j == 0 ? d-k : 1 for j in IdentityRange(0:k)]
    β = [j == 0 ? UYl0/nl : UYl[j]/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    J0 = findall(iszero,γ)
    αtl = sum(α[j] for j in J0)
    βtl = sum(β[j] for j in J0)
    ζtl = sum(α[j]/(γ[j]+vl) for j in 0:k if j ∉ J0)
    B = βtl + sum(β[j]*vl^2/(γ[j]+vl)^2 for j in 0:k if j ∉ J0)

    return (-αtl + sqrt(αtl^2 + 4*ζtl*B))/(2*ζtl)
end

# Update method: Cubic solvable minorizer
"""
    updatevl(vl,U,λ,Yl,::CubicSolvableMinorizer)

Minorize maximize update of `vl` with cubic solvable minorizer.
"""
function updatevl(vl,U,λ,Yl,::CubicSolvableMinorizer)
    d, k = size(U)
    nl = size(Yl,2)

    # Compute coefficients
    UYl = [sum(abs2,Yl'U[:,j]) for j in 1:k]
    UYl0 = nonnegative(sum(abs2,Yl)-sum(UYl))
    α = [j == 0 ? d-k : 1 for j in IdentityRange(0:k)]
    β = [j == 0 ? UYl0/nl : UYl[j]/nl for j in IdentityRange(0:k)]
    γ = [j == 0 ? zero(eltype(λ)) : λ[j] for j in IdentityRange(0:k)]
    c = [-2*β[j]/γ[j]^3 for j in IdentityRange(0:k)]
    J0 = findall(iszero,γ)
    αtl = sum(α[j] for j in J0)
    βtl = sum(β[j] for j in J0)
    ζtl = sum(α[j]/(γ[j]+vl) for j in 0:k if j ∉ J0)
    γtl = -ζtl + sum(β[j]/(γ[j]+vl)^2 for j in 0:k if j ∉ J0)
    ctl = sum(c[j] for j in 0:k if j ∉ J0)

    complexroots = PolynomialRoots.solve_cubic_eq(complex.([βtl,-αtl,γtl-ctl*vl,ctl]))
    vcritical = [real(v) for v in complexroots if real(v) ≈ v && real(v) >= zero(vl)]

    isempty(vcritical) && return zero(vl)
    return _argmax(vcritical) do v
        -αtl*log(v) - βtl/v - ζtl*v + sum(β[j]/(γ[j]+vl)^2*v + (1/2)*c[j]*(v-vl)^2 for j in 0:k if j ∉ J0)
    end
end

# Utilities
nonnegative(x) = max(zero(x),x)
