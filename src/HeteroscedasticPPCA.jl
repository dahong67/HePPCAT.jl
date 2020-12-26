"Heteroscedastic PPCA module. Provides probabilistic PCA for data with heterogeneous quality (heteroscedastic noise)."
module HeteroscedasticPPCA

# Imports
using Base: IdentityUnitRange   # todo: swap with IdentityRanges.IdentityRange if working (https://github.com/JuliaArrays/IdentityRanges.jl/issues/12)
using IdentityRanges: IdentityRange
using IntervalArithmetic: interval, mid
using IntervalRootFinding: Newton, roots
using LinearAlgebra: Diagonal, I, opnorm, norm, qr, svd, /
using Logging: @warn
import PolynomialRoots
using Roots: find_zero
using Statistics: mean

# Exports
export HetPPCA, hetppca, loglikelihood

# More convenient form of argmax not yet available
# + findmax from https://github.com/cmcaine/julia/blob/argmax-2-arg-harder/base/reduce.jl#L704-L705
# + argmax from https://github.com/cmcaine/julia/blob/argmax-2-arg-harder/base/reduce.jl#L830
# + part of pull request https://github.com/JuliaLang/julia/pull/35316
_findmax(f, domain) = mapfoldl(x -> (f(x), x), _rf_findmax, domain)
_rf_findmax((fm, m), (fx, x)) = isless(fm, fx) ? (fx, x) : (fm, m)
_argmax(f, domain) = _findmax(f, domain)[2]

include("types.jl")
include("extern.jl")
include("updatev.jl")
include("updateF.jl")
include("updateU.jl")
include("updatelambda.jl")

end
