# HeteroscedasticPCA

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://github.com/dahong67/HeteroscedasticPCA.jl/workflows/CI/badge.svg)](https://github.com/dahong67/HeteroscedasticPCA.jl/actions)
[![Coverage](https://codecov.io/gh/dahong67/HeteroscedasticPCA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dahong67/HeteroscedasticPCA.jl)

## TODO

+ [x] Copy implementations from previous code and get them running
+ [ ] Changes without numerical impact
  + [x] Initial refactor using cleaner and more uniform signatures for updates (and make them mutating)
  + [x] Some initial tests (that test `==` not `≈`) and benchmarks
  + [ ] Fix inconsistencies related to `θ` vs `θ2`
  + [ ] Try to further simplify the implementations but without changing the outputs
+ [ ] Changes with potential numerical impact
  + [ ] Remove old root-finding method `:oldflatroots`
  + [ ] Simplify implementations but now allowing for changes to the outputs up to precision type stuff
  + [ ] New global maximization ideas for `v` and `θ` updates
+ [ ] Further enhancements
  + [ ] Add all methods from paper
  + [ ] Make reference implementations for tests simple / easily checkable
  + [ ] Simplify/unify tests and benchmarks
  + [ ] Add docs
  + [ ] Improve speed (#ebaa839 to #51326d4 had significant regressions for flat variant)
    + Need to profile and understand where/why its slow
    + Maybe add specific implementations for this case
+ [ ] Run through `JuliaFormatter`
