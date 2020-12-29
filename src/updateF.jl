## F updates

"""
    updateF!(M::HePPCAT,Y,method)

Update the factor matrix in `M` with respect to data `Y` using `method`.
"""
function updateF! end

# Update method: Expectation Maximization
"""
    updateF!(M::HePPCAT,Y,::ExpectationMaximization)

Expectation maximization update of `F`.
"""
function updateF!(M::HePPCAT,Y,::ExpectationMaximization)
    n, L = size.(Y,2), length(Y)
    Λ = Diagonal(M.λ)
    Γ = [inv(Λ + M.v[l]*I) for l in 1:L]
    Z = [Γ[l]*sqrt(Λ)*M.U'*Y[l] for l in 1:L]
    num = sum(Y[l]*Z[l]'/M.v[l] for l in 1:L)
    den = sum(Z[l]*Z[l]'/M.v[l] + n[l]*Γ[l] for l in 1:L)

    F = svd((num / den) * M.Vt)
    M.U .= F.U
    M.λ .= F.S.^2
    M.Vt .= F.Vt
    return M
end
