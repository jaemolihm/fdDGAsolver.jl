# fdDGAsolver

[![Build Status](https://github.com/jaemolihm/fdDGAsolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jaemolihm/fdDGAsolver.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/jaemolihm/fdDGAsolver.jl/graph/badge.svg?token=38YPJVWVMA)](https://codecov.io/gh/jaemolihm/fdDGAsolver.jl)


### TODO
* In `ParquetSolver`, `S.FL` have zero K1 vertices, so evaluations of K1 can be avoided.
* Make mesh_index(SVector, BrillouinMesh) non allocating?
* `euclidean(k, mK)` allocates because `basis(mK)` is type untstable (should be `SMatrix{2,2,F,4}` not `SMatrix{2,2,F}`.) I defined `euclidean` in `matsubarafunctions_piracy.jl` as a workaround.

* In SDE, separate K1-like term in L (U * Pi * K1+K2) and the K2-like term. The former can be summed over wider \nu.
* Store \nu-integrated bubble and use it for K1-Pi-K1 contraction?
