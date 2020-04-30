# PolynomialRatio: minimal implementation based on example from the manual
# https://docs.julialang.org/en/v1/manual/constructors/#Case-Study:-Rational-1
# and base implementation of Rational
# https://github.com/JuliaLang/julia/blob/v1.3.1/base/rational.jl
# May also be worth considering https://github.com/giordano/PolynomialRoots.jl
# for faster and more precise root-finding

module PolynomialRatios

export PolynomialRatio

using Polynomials
import Base: //, show, numerator, denominator, +, -

struct PolynomialRatio{T<:Poly}
    num::T
    den::T
    function PolynomialRatio{T}(num::T, den::T) where T<:Poly
        if num == 0 && den == 0
            error("invalid rational: 0//0")
        end
        # g = gcd(den, num)   # Don't reduce, since it seems to sometimes lose solutions
        # num = div(num, g)
        # den = div(den, g)
        # s = coeffs(num)[end]  # Scale so that leading coefficient is always one
        # num /= s
        # den /= s
        new(num, den)
    end
end
PolynomialRatio(n::T, d::T) where {T<:Poly} = PolynomialRatio{T}(n,d)
PolynomialRatio(n::Poly, d::Poly) = PolynomialRatio(promote(n,d)...)
PolynomialRatio(n::Poly) = PolynomialRatio(n,one(n))

# Based on https://github.com/JuliaLang/julia/blob/v1.3.1/base/rational.jl#L30-L44
# Slight (acceptable?) type-piracy
//(n::Poly,d::Poly) = PolynomialRatio(n,d)
//(n::Number,d::Poly) = PolynomialRatio(n*one(d),d)
//(n::Poly,d::Number) = PolynomialRatio(n,d*one(n))

# Based on https://github.com/JuliaLang/julia/blob/v1.3.1/base/rational.jl#L195-L227
numerator(x::PolynomialRatio) = x.num
denominator(x::PolynomialRatio) = x.den

# Based on https://github.com/JuliaLang/julia/blob/v1.3.1/base/rational.jl#L66-L70
function show(io::IO, x::PolynomialRatio)
    show(io, numerator(x))
    print(io, "//")
    show(io, denominator(x))
end

# Based on https://github.com/JuliaLang/julia/blob/v1.3.1/base/rational.jl#L25-L28
# and https://github.com/JuliaLang/julia/blob/v1.3.1/base/rational.jl#L253-L261
function divgcd(x::Poly,y::Poly)
    g = gcd(x,y)
    div(x,g), div(y,g)
end
function +(x::PolynomialRatio, y::PolynomialRatio)
    # xd, yd = divgcd(x.den, y.den) # Don't reduce, since it seems to sometimes lose solutions
    xd, yd = x.den, y.den
    PolynomialRatio(x.num*yd+y.num*xd, x.den*yd)
end
function -(x::PolynomialRatio, y::PolynomialRatio)
    # xd, yd = divgcd(x.den, y.den) # Don't reduce, since it seems to sometimes lose solutions
    xd, yd = x.den, y.den
    PolynomialRatio(x.num*yd-y.num*xd, x.den*yd)
end

end
