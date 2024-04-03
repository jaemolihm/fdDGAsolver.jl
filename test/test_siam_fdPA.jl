using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Test

@testset "SIAM fdPA" begin
    using MPI
    MPI.Init()

    T = 0.1
    U = 1.0
    D = 10.0
    e = -0.3
    Δ = π / 3

    nmax = 8
    nG  = 24nmax
    nΣ  = 24nmax
    nK1 = 12nmax
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    # scPA for the reference point
    S0 = parquet_solver_siam_parquet_approximation(nG, nΣ, nK1, nK2, nK3; e, T, U, Δ, D)
    fdDGAsolver.init_sym_grp!(S0)
    res = fdDGAsolver.solve!(S0; strategy = :scPA, parallel_mode = :threads, verbose = false);

    # scPA for the target point
    Δ_fd = π / 5
    e_fd = 0.5
    D_fd = 20.0
    S_fd = parquet_solver_siam_parquet_approximation(nG, nΣ, nK1, nK2, nK3; e = e_fd, T, U, Δ = Δ_fd, D = D_fd)
    fdDGAsolver.init_sym_grp!(S_fd)
    res = fdDGAsolver.solve!(S_fd; strategy = :scPA, parallel_mode = :threads, verbose = false);

    # fdPA from the reference to the target
    Gbare = fdDGAsolver.siam_bare_Green(meshes(S0.G, 1); e = e_fd, Δ = Δ_fd, D = D_fd)

    S = ParquetSolver(nG, nΣ, nK1, nK2, nK3, Gbare, S0.G, S0.Σ, S0.F)
    fdDGAsolver.init_sym_grp!(S)
    res = fdDGAsolver.solve!(S; strategy = :fdPA, parallel_mode = :threads, verbose = false);

    @test absmax(S.Σ - S_fd.Σ) < 3e-5

    for ch in [:γa, :γp, :γt], class in [:K1, :K2, :K3]
        @test  absmax(getproperty(getproperty(S.F, ch), class)
                    + getproperty(getproperty(S.F0, ch), class)
                    - getproperty(getproperty(S_fd.F, ch), class)) < 2e-3
    end

end
