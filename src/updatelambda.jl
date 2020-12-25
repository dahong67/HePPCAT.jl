# λ updates
ispos(x) = x > zero(x)

function updateλ!(M::HPPCA,Y,method)
    for (j,uj) in enumerate(eachcol(M.U))
        M.λ[j] = updateλj(M.λ[j],uj,M.v,Y,method)
    end
    return M
end
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
function updateλj(λj,uj,v,Y,::ExpectationMaximization)
    n, L = size.(Y,2), length(Y)

    num = sum(norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) for l in 1:L)
    den = sum(λj/(λj+v[l]) * (norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) + n[l]) for l in 1:L)
    return λj * (num/den)^2
end
function updateλj(λj,uj,v,Y,::MinorizeMaximize)
    all(ispos,v) || throw("Minorizer expects positive v. Got: $v")
    n, L = size.(Y,2), length(Y)

    ζ = [norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) + n[l] for l in 1:L]
    num = sum(norm(Y[l]'uj)^2/v[l] * λj/(λj+v[l]) for l in 1:L) * sum(ζ[l]*v[l]/(λj+v[l]) for l in 1:L)
    den = sum(ζ[l]/(λj+v[l]) for l in 1:L)
    return (1/sum(n)) * num / den
end
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
