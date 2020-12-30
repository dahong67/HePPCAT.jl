## λ updates

"""
    updateλ!(M::HePPCATModel,Y,method)

Update the factor eigenvalues `M.λ` with respect to data `Y` using `method`.
Internally calls [`updateλj(_)`](@ref) to update each entry.
"""
function updateλ!(M::HePPCATModel,Y,method)
    for (j,uj) in enumerate(eachcol(M.U))
        M.λ[j] = updateλj(M.λ[j],uj,M.v,Y,method)
    end
    return M
end

"""
    updateλj(λj,uj,v,Y,method)

Compute update for `j`th factor eigenvalue `λj` with respect to data `Y` using `method`.
"""
function updateλj end

# Update method: Global maximization via root-finding
"""
    updateλj(λj,uj,v,Y,::RootFinding)

Root-finding (global maximization) update of `λj`.
"""
function updateλj(λj,uj,v,Y,::RootFinding)
    n, L = size.(Y,2), length(Y)

    # Compute coefficients and root bounds
    β = [norm(uj'Yl)^2 for Yl in Y]
    λmin, λmax = extrema(max.(zero.(β), β./n .- v))

    # Compute roots
    tol = 1e-8  # todo: choose tolerance adaptively
    λmax-λmin < tol && return (λmax+λmin)/2
    λcritical = roots(interval(λmin,λmax),Newton,tol) do λ
        sum(β[l]/(λ+v[l])^2 - n[l]/(λ+v[l]) for l in 1:L)
    end
    isempty(λcritical) && return λmin
    length(λcritical) == 1 && return mid(interval(first(λcritical)))

    # Return maximizer
    return _argmax(mid.(interval.(λcritical))) do λ
        -sum(n[l]*log(λ+v[l]) + β[l]/(λ+v[l]) for l in 1:L)
    end
end

# Update method: Expectation Maximization
"""
    updateλj(λj,uj,v,Y,::ExpectationMaximization)

Expectation maximization update of `λj`.
"""
function updateλj(λj,uj,v,Y,::ExpectationMaximization)
    n, L = size.(Y,2), length(Y)

    num = sum(norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) for l in 1:L)
    den = sum(λj/(λj+v[l]) * (norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) + n[l]) for l in 1:L)
    return λj * (num/den)^2
end

# Update method: Minorize maximize
# using the minorizer from
#   Y. Sun, A. Breloy, P. Babu, D. P. Palomar, F. Pascal, and G. Ginolhac,
#   "Low-complexity algorithms for low rank clutter parameters estimation in radar systems,"
#   IEEE Transactions on Signal Processing, vol. 64, no. 8, pp. 1986–1998, Apr. 2016.
"""
    updateλj(λj,uj,v,Y,::MinorizeMaximize)

Minorize maximize update of `λj` using the minorizer from
> Y. Sun, A. Breloy, P. Babu, D. P. Palomar, F. Pascal, and G. Ginolhac,
> "Low-complexity algorithms for low rank clutter parameters estimation in radar systems,"
> IEEE Transactions on Signal Processing, vol. 64, no. 8, pp. 1986–1998, Apr. 2016.
"""
function updateλj(λj,uj,v,Y,::MinorizeMaximize)
    all(ispos,v) || throw("Minorizer expects positive v. Got: $v")
    n, L = size.(Y,2), length(Y)

    ζ = [norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) + n[l] for l in 1:L]
    num = sum(norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) for l in 1:L) * sum(ζ[l]*v[l]/(λj+v[l]) for l in 1:L)
    den = sum(ζ[l]/(λj+v[l]) for l in 1:L)
    return (1/sum(n)) * num / den
end

# Update method: Difference of concave approach
"""
    updateλj(λj,uj,v,Y,::DifferenceOfConcave)

Difference of concave update of `λj`.
"""
function updateλj(λj,uj,v,Y,::DifferenceOfConcave)
    n, L = size.(Y,2), length(Y)
    
    # Compute coefficients and check edge case
    ytj = [norm(Y[l]'uj)^2 for l in 1:L]
    affslope = -sum(n[l]/(λj+v[l]) for l in 1:L)
    Ltp = λ -> affslope + sum(ytj[l]/(λ+v[l])^2 for l in 1:L if !iszero(ytj[l]))
    if affslope == -Inf || Ltp(zero(λj)) <= zero(λj)
        return zero(λj)
    end

    # Return nonnegative critical point
    tol = 1e-13  # to get a bracketing interval
    λmax = maximum(sqrt(ytj[l]/n[l]*(λj+v[l])) - v[l] for l in 1:L) + tol
    return find_zero(Ltp,(zero(λmax),λmax))
end

# Update method: Quadratic minorizer
"""
    updateλj(λj,uj,v,Y,::QuadraticMinorizer)

Minorize maximize update of `λj` using quadratic minorizer.
"""
function updateλj(λj,uj,v,Y,::QuadraticMinorizer)
    all(ispos,v) || throw("Minorizer expects positive v. Got: $v")
    n, L = size.(Y,2), length(Y)
    
    num = sum(1:L) do l
        ytlj = norm(Y[l]'uj)^2
        n[l]/(λj+v[l]) - ytlj/(λj+v[l])^2
    end
    den = sum(1:L) do l
        ytlj = norm(Y[l]'uj)^2
        ctlj = -2*ytlj/v[l]^3
        ctlj
    end
    
    return max(zero(λj),λj + num/den)
end

# Update method: Quadratic minorizer with optimized curvature
"""
    updateλj(λj,uj,v,Y,::OptimalQuadraticMinorizer)

Minorize maximize update of `λj` using quadratic minorizer with optimized curvature.
"""
function updateλj(λj,uj,v,Y,::OptimalQuadraticMinorizer)
    all(ispos,v) || throw("Minorizer expects positive v. Got: $v")
    n, L = size.(Y,2), length(Y)
    
    num = sum(1:L) do l
        ytlj = norm(Y[l]'uj)^2
        n[l]/(λj+v[l]) - ytlj/(λj+v[l])^2
    end
    den = sum(1:L) do l
        ytlj = norm(Y[l]'uj)^2
        a, b = n[l], ytlj
        f = λ -> a*log(λ+v[l]) + b/(λ+v[l])
        fd = λ -> a/(λ+v[l]) - b/(λ+v[l])^2
        ctlj = (2b <= a*v[l]) ? zero(λj) : -2*(f(zero(λj)) - f(λj) + fd(λj)*λj)/λj^2
        ctlj
    end
    
    return max(zero(λj),λj + num/den)
end

# Utilities
ispos(x) = x > zero(x)