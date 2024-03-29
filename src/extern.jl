## External interface

# Main function
"""
    heppcat(Y,k,iters;init=homppca(Y,k),vknown=false,varfloor=zero(eltype(init.v)))

Estimate probabilistic PCA model for noise that is heteroscedastic across samples.

# Required Inputs
+ `Y` : list of matrices (each column is a sample)
+ `k` : number of factors
+ `iters`  : number of iterations to run

# Optional Keyword Arguments
+ `init`   : initial model (will be modified in-place)
+ `vknown` : variances are known (do not update) default `false`
+ `varfloor` : lower bound for variance iterates (useful if the iterates are degenerating to zero) default = 0

Output is a [`HePPCATModel`](@ref) object.
"""
function heppcat(Y,k,iters::Integer;init=homppca(Y,k),vknown::Bool=false,varfloor=zero(eltype(init.v)))
    M = init
    vmethod = iszero(varfloor) ? ExpectationMaximization() : ProjectedVariance(ExpectationMaximization(),varfloor)
    Fmethod = ExpectationMaximization()
    for _ in 1:iters
        vknown || updatev!(M,Y,vmethod)
        updateF!(M,Y,Fmethod)
    end
    M
end

# Homoscedastic initialization
"""
    homppca(Y,k)

Estimate probabilistic PCA model for noise that is homoscedastic across samples.

Inputs are:
+ `Y` : list of matrices (each column is a sample)
+ `k` : number of factors
Output is a [`HePPCATModel`](@ref) object.
"""
function homppca(Y,k)
    Yf = reduce(hcat,Y)
    n, L = size(Yf,2), length(Y)
    Uh, s, _ = svd(Yf)
    λh = abs2.(s)./n
    λb = mean(λh[k+1:end])
    HePPCATModel(Uh[:,1:k],λh[1:k] .- λb,I(k),fill(λb,L))
end

# log-likelihood (todo: add constant)
"""
    loglikelihood(M::HePPCATModel,Y)

Log-likelihood of model `M` with respect to data `Y` (dropping constant term).
"""
function loglikelihood(M::HePPCATModel,Y)
    d, k = size(M.U)
    n, L = size.(Y,2), length(Y)
    return 1/2*sum(1:L) do l
        norm(sqrt(Diagonal((M.λ./M.v[l])./(M.λ .+ M.v[l])))*M.U'Y[l])^2 -
        n[l]*sum(log.(M.λ .+ M.v[l])) - n[l]*(d-k)*log(M.v[l]) - norm(Y[l])^2/M.v[l]
    end
end
