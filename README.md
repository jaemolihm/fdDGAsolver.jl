# fdDGAsolver

[![Build Status](https://github.com/jaemolihm/fdDGAsolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jaemolihm/fdDGAsolver.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/jaemolihm/fdDGAsolver.jl/graph/badge.svg?token=38YPJVWVMA)](https://codecov.io/gh/jaemolihm/fdDGAsolver.jl)


### API changes made by JML (13.04.2024)
* The `nΣ` argument of `parquet_solver_*` function is removed. `Σ` uses the same grid as `G`, which is determined by `nG`.
* The `nG` and `nΣ` arguments of `ParquetSolver` constructors are removed. The meshes of `Gbare` is used.

### TODO
* Fix disk logging (e.g. per 10 iterations)
* Fix reading Vertex (Currently done using quick-and-dirty try catch)


### TODO (Comments from Dominik)
- [x] Use `buffer_fold_back`
    * Instead of using buffering, I defined `fixed_momentum_view` so that in the NL2 K2 BSE, the momentum indexing occurs only once per q loop. (Previously it was inside both q and omega loops so it occured `N_ω * N_q` times per call to `diagram`. Now its `N_q` times.)
- [ ] Type instability for `init_sym_grp`
- [x] There is a modified add! function in matsubarafunctions_piracy which allows you to multiply with a number and then add to another MeshFunction, would it be helpful to have that in the library? Then we could remove it here.
- [x] Do we need all versions of box_eval? It seems that this is now only very rarely used.
- [x] How is the density spin component defined and what purpose does it serve?
- [x] Would it be nicer to have one function which converts momenta and frequencies?
- [x] The reduce function now has to go through a couple of if statements regarding the max_class parameter? How about we introduce some enum similar as for the SpinTag and ChannelTag that denotes the asymptotic Kernel so we can have a compile time dispatch?
- [ ] The method loading CTINT reference data, is that still working or is it outdated?
- [x] in Dyson! Why do we not iterate over MeshPoints but the values? Is the Sigma grid different from the G grid?
- [ ] Can you remind me of how you ultimately calculate the SDE and maybe document the different contributions?
- [x] Comment the K3 cache in a bit more detail please!
- [x] When using getindex with an SWaveBrillouinPoint the respective momentum argument is summed over. Where and why do we need that?
- [ ] Split the three solvers into different branches
