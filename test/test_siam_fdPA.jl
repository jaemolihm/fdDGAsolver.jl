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
        @test absmax(getproperty(getproperty(S.F, ch), class)
                   + getproperty(getproperty(S.F0, ch), class)
                   - getproperty(getproperty(S_fd.F, ch), class)) < 2e-3
    end

    # fdPA with different box sizes for the vertex
    nmax = 6
    nK1 = 12nmax
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    Gbare = fdDGAsolver.siam_bare_Green(meshes(S0.G, 1); e = e_fd, Δ = Δ_fd, D = D_fd)

    S2 = ParquetSolver(nG, nΣ, nK1, nK2, nK3, Gbare, S0.G, S0.Σ, S0.F)
    fdDGAsolver.init_sym_grp!(S2)
    res = fdDGAsolver.solve!(S2; strategy = :fdPA, parallel_mode = :threads, verbose = false);

    @test absmax(S2.Σ - S_fd.Σ) < 3e-3
    @test S2.Σ(π*T) ≈ 0.024627233663216237 - 0.1752772891026662im

    Ω = MatsubaraFrequency(T, 0, Boson)
    ν = MatsubaraFrequency(T, 1, Fermion)
    νp = MatsubaraFrequency(T, -2, Fermion)
    for ch in [:γa, :γp, :γt]
        z1 = getproperty(S2.F, ch)(Ω, ν, νp) + getproperty(S2.F0, ch)(Ω, ν, νp)
        z2 = getproperty(S_fd.F, ch)(Ω, ν, νp)
        @test z1 ≈ z2 atol = 2e-4
    end
    for ch in [pCh, tCh, aCh], sp in [pSp, xSp]
        @test S2.F(Ω, ν, νp, ch, sp) ≈ S_fd.F(Ω, ν, νp, ch, sp) atol = 4e-4
    end

    # Test Vertex evaluation

    Ω = MatsubaraFrequency(T, 1, Boson)
    ω = MatsubaraFrequency(T, 3, Fermion)

    # If one of the fermionic frequency is νInf, other channels should not be evaluated
    for F0 in (true, false), ch in (pCh, tCh, aCh), sp in (pSp, xSp, dSp)
        x = S.F(Ω, ω, νInf, ch, sp; F0, γp = (ch === pCh), γt = (ch === tCh), γa = (ch === aCh))
        y = S.F(Ω, ω, νInf, ch, sp; F0)
        @test x ≈ y

        x = S.F(Ω, νInf, ω, ch, sp; F0, γp = (ch === pCh), γt = (ch === tCh), γa = (ch === aCh))
        y = S.F(Ω, νInf, ω, ch, sp; F0)
        @test x ≈ y
    end

    @test S.F(Ω, ω, νInf, pCh, pSp) ≈ S.F.γp(Ω, ω, νInf) + S.F.F0.γp(Ω, ω, νInf) + S.F.F0.F0(Ω, ω, νInf, pCh, pSp)
    @test S.F(Ω, ω, νInf, tCh, pSp) ≈ S.F.γt(Ω, ω, νInf) + S.F.F0.γt(Ω, ω, νInf) + S.F.F0.F0(Ω, ω, νInf, tCh, pSp)
    @test S.F(Ω, ω, νInf, aCh, pSp) ≈ S.F.γa(Ω, ω, νInf) + S.F.F0.γa(Ω, ω, νInf) + S.F.F0.F0(Ω, ω, νInf, aCh, pSp)
    @test S.F(Ω, νInf, ω, pCh, pSp) ≈ S.F.γp(Ω, νInf, ω) + S.F.F0.γp(Ω, νInf, ω) + S.F.F0.F0(Ω, νInf, ω, pCh, pSp)
    @test S.F(Ω, νInf, ω, tCh, pSp) ≈ S.F.γt(Ω, νInf, ω) + S.F.F0.γt(Ω, νInf, ω) + S.F.F0.F0(Ω, νInf, ω, tCh, pSp)
    @test S.F(Ω, νInf, ω, aCh, pSp) ≈ S.F.γa(Ω, νInf, ω) + S.F.F0.γa(Ω, νInf, ω) + S.F.F0.F0(Ω, νInf, ω, aCh, pSp)
end
