## External interface

# Main function
function hetppca(Y,k,iters;init=homppca(Y,k))
    M = init
    for _ in 1:iters
        updatev!(M,Y,ExpectationMaximization())
        updateF!(M,Y,ExpectationMaximization())
    end
    M
end

# Homoscedastic initialization
function homppca(Y,k)
    Yf = reduce(hcat,Y)
    n, L = size(Yf,2), length(Y)
    Uh, s, _ = svd(Yf)
    λh = abs2.(s)./n
    λb = mean(λh[k+1:end])
    HetPPCA(Uh[:,1:k],λh[1:k] .- λb,I(k),fill(λb,L))
end

# log-likelihood (todo: add constant)
function loglikelihood(M,Y)
    d, k = size(M.U)
    n, L = size.(Y,2), length(Y)
    return 1/2*sum(1:L) do l
        norm(sqrt(Diagonal((M.λ./M.v[l])./(M.λ .+ M.v[l])))*M.U'Y[l])^2 -
        n[l]*sum(log.(M.λ .+ M.v[l])) - n[l]*(d-k)*log(M.v[l]) - norm(Y[l])^2/M.v[l]
    end
end
