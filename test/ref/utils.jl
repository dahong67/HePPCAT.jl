module Utils

using Polynomials: Poly, polyder, degree
using IntervalArithmetic: @interval, mid, interval
using IntervalRootFinding: roots, Newton

export posroots

## Positive root-finding
function posroots(p::Poly{T}, tol::T) where {T}
    bound = _hongbound(p)
    bound < tol && return [tol/2]::Vector{T}

    _roots = roots(p, polyder(p), @interval(zero(bound),bound), Newton, tol)
    return [mid(interval(r)) for r in _roots]::Vector{T}
end

# doi: 10.1006/jsco.1997.0189
function _hongbound(f::Poly{T}) where {T}
    d = degree(f)
    a = sign(f[d]) * f    # make leading coefficient positive
    return 2*maximum(filter(q -> a[q] < zero(a[q]), 0:d)) do q
        minimum(filter(p -> a[p] > zero(a[p]), q+1:d)) do p
            abs(a[q]/a[p])^(1/(p-q))
        end
    end::T
end

end
