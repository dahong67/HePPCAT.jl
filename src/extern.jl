# PPCA
function ppca(Y,k,iters,init,::Val{:sage})
    M = HPPCA(svd(init).U,svd(init).S.^2,svd(init).Vt,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        updateF!(M,Y,ExpectationMaximization())
        push!(MM, deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM, :λ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:mm})
    M = HPPCA(svd(init).U,svd(init).S.^2,svd(init).Vt,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        updateλ!(M,Y,RootFinding())
        updateU!(M,Y,MinorizeMaximize())
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:λ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:pgd})
    Ynorms = vec(mapslices(norm,hcat(Y...),dims=1))
    M = HPPCA(svd(init).U,svd(init).S.^2,svd(init).Vt,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        updateλ!(M,Y,RootFinding())
        L = sum(ynorm^2*maximum([λj/vi/(λj+vi) for λj in M.λ])
            for (ynorm,vi) in zip(Ynorms,M.v))
        updateU!(M,Y,ProjectedGradientAscent(1/L))
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:λ), getfield.(MM,:v)
end
function ppca(Y,k,iters,init,::Val{:sgd},max_line=50,α=0.8,β=0.5,σ=1.0)
    M = HPPCA(svd(init).U,svd(init).S.^2,svd(init).Vt,zeros(length(Y)))
    MM = [deepcopy(M)]
    for t = 1:iters
        updatev!(M,Y,RootFinding())
        updateλ!(M,Y,RootFinding())
        updateU!(M,Y,StiefelGradientAscent(ArmijoSearch(max_line,α,β,σ)))
        push!(MM,deepcopy(M))
    end
    return getfield.(MM,:U), getfield.(MM,:λ), getfield.(MM,:v)
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
