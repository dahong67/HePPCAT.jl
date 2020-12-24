# HeteroscedasticPCA (todo: rename?)

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://github.com/dahong67/HeteroscedasticPCA.jl/workflows/CI/badge.svg)](https://github.com/dahong67/HeteroscedasticPCA.jl/actions)
[![Coverage](https://codecov.io/gh/dahong67/HeteroscedasticPCA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dahong67/HeteroscedasticPCA.jl)

+ [ ] Further enhancements
  + [ ] Add missing approaches: PGA line search, StGA constant step
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
+ [ ] Run through `JuliaFormatter`
+ [ ] Test edge cases for the updates (e.g., cases where objective may evaluate to `+Inf`, `-Inf` or `NaN` or where root-finding may encounter issues)
+ [ ] Test convergence to critical points
+ [ ] Maybe have `LipBoundU1` and `LipBoundU2` take data and output a function mapping an iterate to Lipschitz bound (might be cleaner and allows for pre-computation)
+ [ ] **Rename to something like `HeterogeneousPCA.jl`**
+ [ ] Release as public and register!
+ [ ] Put data in update method objects to unify and speed up?
+ [ ] Add/test minorizers?
+ [ ] Test that MM updates maximize the minorizer by optimizing minorizer with off-the-shelf optimizer? (maybe use that as the reference implementation?)