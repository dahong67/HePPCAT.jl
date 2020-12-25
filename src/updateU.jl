# U updates
function polar(A)
    F = svd(A)
    return F.U*F.Vt
end
gradF(U,λ,v,Y) = sum(Yl * Yl' * U * Diagonal(λ./vl./(λ.+vl)) for (Yl,vl) in zip(Y,v))
F(U,λ,v,Y) = 1/2*sum(norm(sqrt(Diagonal(λ./vl./(λ.+vl)))*U'*Yl)^2 for (Yl,vl) in zip(Y,v))
function LipBoundU1(M::HPPCA,Y)
    L, λmax = length(M.v), maximum(M.λ)
    return sum(norm(Y[l])^2*λmax/M.v[l]/(λmax+M.v[l]) for l in 1:L)
end
function LipBoundU2(M::HPPCA,Y)
    L, λmax = length(M.v), maximum(M.λ)
    return sum(opnorm(Y[l])^2*λmax/M.v[l]/(λmax+M.v[l]) for l in 1:L)
end

updateU!(M::HPPCA,Y,::MinorizeMaximize) = (M.U .= polar(gradF(M.U,M.λ,M.v,Y)); M)
function updateU!(M::HPPCA,Y,pga::ProjectedGradientAscent{<:Number})
    if pga.stepsize == Inf
        M.U .= polar(gradF(M.U,M.λ,M.v,Y))
    else
        M.U .= polar(M.U + pga.stepsize*gradF(M.U,M.λ,M.v,Y))
    end
    return M
end
updateU!(M::HPPCA,Y,pga::ProjectedGradientAscent{<:InverseLipschitz}) =
    updateU!(M,Y,ProjectedGradientAscent(inv(pga.stepsize.bound(M,Y))))
function updateU!(M::HPPCA,Y,pga::ProjectedGradientAscent{<:ArmijoSearch})
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
function updateU!(M::HPPCA,Y,sga::StiefelGradientAscent{<:ArmijoSearch})
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
skew(A) = (A-A')/2
function geodesic(U,X,t)
    k = size(U,2)

    A = skew(U'X)
    Q,R = qr(X - U*(U'X))

    MN = exp(t*[A -R'; R zeros(k,k)])[:,1:k]
    M, N = MN[1:k,:], MN[k+1:end,:]

    return U*M + Matrix(Q)*N
end
