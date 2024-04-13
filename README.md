# fdDGAsolver

[![Build Status](https://github.com/jaemolihm/fdDGAsolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jaemolihm/fdDGAsolver.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/jaemolihm/fdDGAsolver.jl/graph/badge.svg?token=38YPJVWVMA)](https://codecov.io/gh/jaemolihm/fdDGAsolver.jl)


### TODO
* In `ParquetSolver`, `S.FL` have zero K1 vertices, so evaluations of K1 can be avoided.
* Make mesh_index(SVector, BrillouinMesh) non allocating?
* `euclidean(k, mK)` allocates because `basis(mK)` is type untstable (should be `SMatrix{2,2,F,4}` not `SMatrix{2,2,F}`.) I defined `euclidean` in `matsubarafunctions_piracy.jl` as a workaround.


### TODO (Comments from Dominik)
- [ ] Use `buffer_fold_back`
- [ ] Type instability for `init_sym_grp`
- [x] There is a modified add! function in matsubarafunctions_piracy which allows you to multiply with a number and then add to another MeshFunction, would it be helpful to have that in the library? Then we could remove it here.
- [ ] Do we need all versions of box_eval? It seems that this is now only very rarely used.
- [ ] How is the density spin component defined and what purpose does it serve?
- [ ] Would it be nicer to have one function which converts momenta and frequencies?
- [ ] The reduce function now has to go through a couple of if statements regarding the max_class parameter? How about we introduce some enum similar as for the SpinTag and ChannelTag that denotes the asymptotic Kernel so we can have a compile time dispatch?
- [ ] The method loading CTINT reference data, is that still working or is it outdated?
- [ ] in Dyson! Why do we not iterate over MeshPoints but the values? Is the Sigma grid different from the G grid?
- [ ] Can you remind me of how you ultimately calculate the SDE and maybe document the different contributions?
- [ ] Comment the K3 cache in a bit more detail please!
- [ ] When using getindex with an SWaveBrillouinPoint the respective momentum argument is summed over. Where and why do we need that?
- [ ] Split the three solvers into different branches
