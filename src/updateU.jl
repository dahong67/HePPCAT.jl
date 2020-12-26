## U updates

"""
    updateU!(M::HetPPCA,Y,method)

Update the factor eigenvectors `M.U` with respect to data `Y` using `method`.
"""
function updateU! end

# Update method: Minorize Maximize (with linear minorizer)
"""
    updateU!(M::HetPPCA,Y,::MinorizeMaximize)

Minorize maximize update of `M.U` using a linear minorizer.
"""
function updateU!(M::HetPPCA,Y,::MinorizeMaximize)
    M.U .= polar(gradF(M.U,M.λ,M.v,Y))
    return M
end

# Update method: Projected Gradient Ascent
"""
    updateU!(M::HetPPCA,Y,pga::ProjectedGradientAscent)

Projected gradient ascent update of `M.U`.
Supported step size types:
`Number` (i.e., constant step size),
[`InverseLipschitz`](@ref),
and
[`ArmijoSearch`](@ref) (experimental).
"""
function updateU!(M::HetPPCA,Y,pga::ProjectedGradientAscent{<:Number})
    if pga.stepsize == Inf
        M.U .= polar(gradF(M.U,M.λ,M.v,Y))
    else
        M.U .= polar(M.U + pga.stepsize*gradF(M.U,M.λ,M.v,Y))
    end
    return M
end
function updateU!(M::HetPPCA,Y,pga::ProjectedGradientAscent{<:InverseLipschitz})
    return updateU!(M,Y,ProjectedGradientAscent(inv(pga.stepsize.bound(M,Y))))
end
function updateU!(M::HetPPCA,Y,pga::ProjectedGradientAscent{<:ArmijoSearch})
    params = pga.stepsize
    
    dFdU = gradF(M.U,M.λ,M.v,Y)
    F0, FΔ = F(M.U,M.λ,M.v,Y), params.tol * norm(dFdU)^2
    m = findfirst(IdentityUnitRange(0:params.maxsearches-1)) do m
        Δ = params.contraction^m * params.stepsize
        F(polar(M.U + Δ*dFdU),M.λ,M.v,Y) >= F0 + Δ * FΔ
    end
    if isnothing(m)
        @warn "Exceeded maximum line search iterations. Accuracy not guaranteed."
        m = params.maxsearches
    end

    Δ = params.contraction^m * params.stepsize
    M.U .= polar(M.U + Δ*dFdU)
    return M
end

# Update method: Stiefel Gradient Ascent
"""
    updateU!(M::HetPPCA,Y,sga::StiefelGradientAscent)

Stiefel gradient ascent update of `M.U`.
Supported step size types:
[`ArmijoSearch`](@ref).
"""
function updateU!(M::HetPPCA,Y,sga::StiefelGradientAscent{<:ArmijoSearch})
    params = sga.stepsize
    
    dFdU = gradF(M.U,M.λ,M.v,Y)
    ∇F = dFdU - M.U*(dFdU'M.U)

    F0, FΔ = F(M.U,M.λ,M.v,Y), params.tol * norm(∇F)^2
    m = findfirst(IdentityUnitRange(0:params.maxsearches-1)) do m
        Δ = params.contraction^m * params.stepsize
        F(geodesic(M.U,∇F,Δ),M.λ,M.v,Y) >= F0 + Δ * FΔ
    end
    if isnothing(m)
        @warn "Exceeded maximum line search iterations. Accuracy not guaranteed."
        m = params.maxsearches
    end

    Δ = params.contraction^m * params.stepsize
    M.U .= geodesic(M.U,∇F,Δ)
    return M
end

# Objective with respect to U
"""
    gradF(U,λ,v,Y)

Euclidean gradient of objective for the optimization problem w.r.t `U`.
"""
gradF(U,λ,v,Y) = sum(Yl * Yl' * U * Diagonal(λ./vl./(λ.+vl)) for (Yl,vl) in zip(Y,v))

"""
    F(U,λ,v,Y)

Objective for the optimization problem w.r.t `U`.
"""
F(U,λ,v,Y) = 1/2*sum(norm(sqrt(Diagonal(λ./vl./(λ.+vl)))*U'*Yl)^2 for (Yl,vl) in zip(Y,v))

"""
    LipBoundU1(M::HetPPCA,Y)

Lipschitz bound of objective w.r.t. `U` at `M` with data `Y`.
"""
function LipBoundU1(M::HetPPCA,Y)
    L, λmax = length(M.v), maximum(M.λ)
    return sum(norm(Y[l])^2*λmax/M.v[l]/(λmax+M.v[l]) for l in 1:L)
end

"""
    LipBoundU2(M::HetPPCA,Y)

Lipschitz bound of objective w.r.t. `U` at `M` with data `Y`.
"""
function LipBoundU2(M::HetPPCA,Y)
    L, λmax = length(M.v), maximum(M.λ)
    return sum(opnorm(Y[l])^2*λmax/M.v[l]/(λmax+M.v[l]) for l in 1:L)
end

# Utilities
"""
    polar(A)

Polar factor of `A`.
"""
function polar(A)
    F = svd(A)
    return F.U*F.Vt
end

"""
    skew(A)

Skew-symmetrize `A` by computing `(A-A')/2`.
"""
skew(A) = (A-A')/2

"""
    geodesic(U,X,t)

Geodesic step of size `t` from `U` in direction of `X` along the Stiefel manifold.
"""
function geodesic(U,X,t)
    k = size(U,2)

    A = skew(U'X)
    Q,R = qr(X - U*(U'X))

    MN = exp(t*[A -R'; R zeros(k,k)])[:,1:k]
    M, N = MN[1:k,:], MN[k+1:end,:]

    return U*M + Matrix(Q)*N
end
