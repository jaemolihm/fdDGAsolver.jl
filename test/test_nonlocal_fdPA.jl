using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using Test

@testset "NL Hubbard fdPA" begin
    using MPI
    MPI.Init()

    T = 0.5
    U = 2.0
    μ = 0.0
    t1 = 1.0

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    nmax = 2
    nG  = 12nmax
    nΣ  = 12nmax
    nK1 = 8nmax
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    nq = 3
    mK_G = BrillouinZoneMesh(BrillouinZone(24, k1, k2))
    mK_Γ = BrillouinZoneMesh(BrillouinZone(nq, k1, k2))

    # Reference solution
    S0 = parquet_solver_hubbard_parquet_approximation(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1, mode = :threads)
    init_sym_grp!(S0)
    res = fdDGAsolver.solve!(S0; strategy = :scPA, verbose = false)


    # Test scPA and fdPA gives identical results for the trivial case (G0 = Σ0 = Π0 = 0)
    S0_fd = parquet_solver_hubbard_parquet_approximation(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1, mode = :threads)
    init_sym_grp!(S0_fd)
    res = fdDGAsolver.solve!(S0_fd; strategy = :fdPA, verbose = false);

    @test absmax(S0_fd.Σ - S0.Σ) < 1e-10
    @test maximum(abs.(flatten(S0_fd.F) .- flatten(S0.F))) < 1e-10


    # Target solution at μ_fd and t2_fd
    μ_fd = 0.5
    t2_fd = -0.3
    S = parquet_solver_hubbard_parquet_approximation(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ = μ_fd, t1, t2 = t2_fd, mode = :threads)
    init_sym_grp!(S)
    res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false)


    # Solve target using fdPA
    Gbare = hubbard_bare_Green(meshes(S0.G)...; μ = μ_fd, t1, t2 = t2_fd)
    S_fd = NL_ParquetSolver(nG, nΣ, nK1, nK2, nK3, mK_Γ, Gbare, S0.G, S0.Σ, S0.F; mode = :threads)
    init_sym_grp!(S_fd)
    fdDGAsolver.solve!(S_fd; strategy = :fdPA, verbose = false)


    # Test fdPA self-energy
    @test absmax(S_fd.Σ - S.Σ) < 2e-4

    # Test fdPA vertex
    Ω  = MatsubaraFrequency(T, 0, Boson)
    ν  = MatsubaraFrequency(T, 1, Fermion)
    νp = MatsubaraFrequency(T, -2, Fermion)
    P  = value(mK_Γ[2])
    k  = value(mK_Γ[3])
    kp = value(mK_Γ[4])

    for ch in [pCh, tCh, aCh], sp in [pSp, xSp]
        @test S_fd.F(Ω, ν, νp, P, k, kp, ch, sp) ≈ S.F(Ω, ν, νp, P, k, kp, ch, sp) atol = 2e-3
    end
end


@testset "NL2 Hubbard fdPA" begin
    using MPI
    MPI.Init()

    T = 0.5
    U = 2.0
    μ = 0.0
    t1 = 1.0

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    nmax = 2
    nG  = 12nmax
    nΣ  = 12nmax
    nK1 = 8nmax
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    nq = 3
    mK_G = BrillouinZoneMesh(BrillouinZone(3, k1, k2))
    mK_Γ = BrillouinZoneMesh(BrillouinZone(nq, k1, k2))

    # Reference solution
    S0 = parquet_solver_hubbard_parquet_approximation_NL2(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1, mode = :threads)
    init_sym_grp!(S0)
    res = fdDGAsolver.solve!(S0; strategy = :scPA, verbose = false)


    # Test scPA and fdPA gives identical results for the trivial case (G0 = Σ0 = Π0 = 0)
    S0_fd = parquet_solver_hubbard_parquet_approximation_NL2(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1, mode = :threads)
    init_sym_grp!(S0_fd)
    res = fdDGAsolver.solve!(S0_fd; strategy = :fdPA, verbose = false);

    @test absmax(S0_fd.Σ - S0.Σ) < 1e-10
    @test maximum(abs.(flatten(S0_fd.F) .- flatten(S0.F))) < 1e-10


    # Target solution at μ_fd and t2_fd
    μ_fd = 0.5
    t2_fd = -0.3
    S = parquet_solver_hubbard_parquet_approximation_NL2(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ = μ_fd, t1, t2 = t2_fd, mode = :threads)
    init_sym_grp!(S)
    res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false)


    # Solve target using fdPA
    Gbare = hubbard_bare_Green(meshes(S0.G)...; μ = μ_fd, t1, t2 = t2_fd)
    S_fd = NL2_ParquetSolver(nG, nΣ, nK1, nK2, nK3, mK_Γ, Gbare, S0.G, S0.Σ, S0.F; mode = :threads)
    init_sym_grp!(S_fd)
    fdDGAsolver.solve!(S_fd; strategy = :fdPA, verbose = false)


    # Test fdPA self-energy
    @test absmax(S_fd.Σ - S.Σ) < 2e-4

    # Test fdPA vertex
    Ω  = MatsubaraFrequency(T, 0, Boson)
    ν  = MatsubaraFrequency(T, 1, Fermion)
    νp = MatsubaraFrequency(T, -2, Fermion)
    P  = value(mK_Γ[2])
    k  = value(mK_Γ[3])
    kp = value(mK_Γ[4])

    for ch in [pCh, tCh, aCh], sp in [pSp, xSp]
        @test S_fd.F(Ω, ν, νp, P, k, kp, ch, sp) ≈ S.F(Ω, ν, νp, P, k, kp, ch, sp) atol = 2e-3
    end
end
