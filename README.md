# HeteroscedasticPCA

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://github.com/dahong67/HeteroscedasticPCA.jl/workflows/CI/badge.svg)](https://github.com/dahong67/HeteroscedasticPCA.jl/actions)
[![Coverage](https://codecov.io/gh/dahong67/HeteroscedasticPCA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dahong67/HeteroscedasticPCA.jl)

## TODO

+ [x] Copy implementations from previous code and get them running
+ [x] Changes without numerical impact
  + [x] Initial refactor using cleaner and more uniform signatures for updates (and make them mutating)
  + [x] Some initial tests (that test `==` not `≈`) and benchmarks
  + [x] Fix inconsistencies related to `θ` vs `θ2`
  + [x] Try to further simplify the implementations but without changing the outputs
+ [x] Changes with potential numerical impact
  + [x] Remove old root-finding method `:oldflatroots`
  + [x] Relax tests by testing updates individually on a few sets of iterates; testing the full iterative method may be too strict since numerical differences may accumulate.
  + [x] Make all updates handle blocks correctly
  + [x] Simplify implementations but now allowing for changes to the outputs up to precision type stuff
  + [x] Work on `updateF!` - ~~can `Vt` be dropped?~~ decided to instead include `Vt` in the `HPPCA` type
  + [x] New global maximization ideas for `v` and `θ` updates
  + [x] ~~Use `geodesic` method from `Manifolds.jl`~~ (`Manifolds.jl` appears to have the "embedded geodesic" whereas we are using the one from the "canonical metric" discussed in `edelman1998tgo`)
+ [ ] Further enhancements
  + [x] Give updates meaningful return values
  + [ ] Add all methods from paper
  + [ ] Add missing approaches: PGA line search, StGA constant step
  + [ ] Add tests to cover more cases (e.g., blocks vs. flat)
  + [ ] Make reference implementations for tests simple / easily checkable
  + [ ] Simplify/unify tests and benchmarks
  + [ ] Add docs
  + [ ] Improve speed
    + #ebaa839 to #51326d4 had significant regressions for flat variant
    + #a6837d9 to #f5663c2 also introduced significant regressions across the board (question: does it have better numerical behavior though?)
      + some attempts that didn't work:
        + using `one` in hopes of encouraging type stability
        + using `zip` in hopes of discouraging bounds checking
        + creating a callable `RationalSum` type in hopes of discouraging unnecessary specializations of `roots`
        + skipping root-finding if the range is already less than the tolerance
        + using `Bisection` instead of `Newton`
      + some things to check/try
        + profile to see where time is spent
        + try version that clears the denominator but does not expand (i.e., don't form a polynomial in the canonical basis) - does the interval method somehow converge more slowly here due to the division?
    + Need to profile and understand where/why its slow
    + Maybe add specific implementations for this case
  + [ ] Consider contributing code for the canonical metric to `Manifolds.jl`
+ [ ] Run through `JuliaFormatter`
+ [ ] Test edge cases for the updates (e.g., cases where objective may evaluate to `+Inf`, `-Inf` or `NaN` or where root-finding may encounter issues)
+ [ ] Test convergence to critical points
