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
    nK1 = 12nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    # scPA for the reference point
    S0 = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, U, Δ, D)
    init_sym_grp!(S0)
    res = fdDGAsolver.solve!(S0; strategy = :scPA, parallel_mode = :threads, verbose = false);


    # fdPA should run and converge to the same point (trivial case of G0 = Σ0 = Π0 = 0)
    S0_fd = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, U, Δ, D)
    init_sym_grp!(S0_fd)
    res = fdDGAsolver.solve!(S0_fd; strategy = :fdPA, parallel_mode = :threads, verbose = false);
    @test absmax(S0.Σ - S0_fd.Σ) < 1e-10
    @test maximum(abs.(flatten(S0.F) .- flatten(S0_fd.F))) < 1e-10


    # scPA for the target point
    Δ_fd = π / 5
    e_fd = 0.5
    D_fd = 20.0
    S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e = e_fd, T, U, Δ = Δ_fd, D = D_fd)
    init_sym_grp!(S)
    res = fdDGAsolver.solve!(S; strategy = :scPA, parallel_mode = :threads, verbose = false);


    # fdPA from the reference to the target
    Gbare = fdDGAsolver.siam_bare_Green(meshes(S0.G, Val(1)); e = e_fd, Δ = Δ_fd, D = D_fd)

    S_fd = ParquetSolver(nK1, nK2, nK3, Gbare, S0.G, S0.Σ, S0.F)
    init_sym_grp!(S_fd)
    res = fdDGAsolver.solve!(S_fd; strategy = :fdPA, parallel_mode = :threads, verbose = false);

    @test absmax(S_fd.Σ - S.Σ) < 3e-5

    @test absmax(S_fd.F.γp.K1 + S_fd.F0.γp.K1 - S.F.γp.K1) < 5e-4
    @test absmax(S_fd.F.γa.K1 + S_fd.F0.γa.K1 - S.F.γa.K1) < 5e-4
    @test absmax(S_fd.F.γt.K1 + S_fd.F0.γt.K1 - S.F.γt.K1) < 5e-4
    @test absmax(S_fd.F.γp.K2 + S_fd.F0.γp.K2 - S.F.γp.K2) < 1e-3
    @test absmax(S_fd.F.γa.K2 + S_fd.F0.γa.K2 - S.F.γa.K2) < 1e-3
    @test absmax(S_fd.F.γt.K2 + S_fd.F0.γt.K2 - S.F.γt.K2) < 1e-3
    @test absmax(S_fd.F.γp.K3 + S_fd.F0.γp.K3 - S.F.γp.K3) < 1e-3
    @test absmax(S_fd.F.γa.K3 + S_fd.F0.γa.K3 - S.F.γa.K3) < 1e-3
    @test absmax(S_fd.F.γt.K3 + S_fd.F0.γt.K3 - S.F.γt.K3) < 1e-3

    for ch in [:γa, :γp, :γt], class in [:K1, :K2, :K3]
        @test absmax(getproperty(getproperty(S_fd.F, ch), class)
                   + getproperty(getproperty(S_fd.F0, ch), class)
                   - getproperty(getproperty(S.F, ch), class)) < 2e-3
    end


    # fdPA with different box sizes for the vertex
    nmax = 6
    nK1 = 12nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    Gbare = fdDGAsolver.siam_bare_Green(meshes(S0.G, Val(1)); e = e_fd, Δ = Δ_fd, D = D_fd)

    S_fd2 = ParquetSolver(nK1, nK2, nK3, Gbare, S0.G, S0.Σ, S0.F)
    init_sym_grp!(S_fd2)
    res = fdDGAsolver.solve!(S_fd2; strategy = :fdPA, parallel_mode = :threads, verbose = false);

    @test absmax(S_fd2.Σ - S.Σ) < 3e-3
    @test S_fd2.Σ(π*T) ≈ 0.02474594503345386 - 0.1761305108665338im

    @test absmax(S_fd.F.γp.K1 + S_fd.F0.γp.K1 - S.F.γp.K1) < 5e-4
    @test absmax(S_fd.F.γa.K1 + S_fd.F0.γa.K1 - S.F.γa.K1) < 5e-4
    @test absmax(S_fd.F.γt.K1 + S_fd.F0.γt.K1 - S.F.γt.K1) < 5e-4
    @test absmax(S_fd.F.γp.K2 + S_fd.F0.γp.K2 - S.F.γp.K2) < 1e-3
    @test absmax(S_fd.F.γa.K2 + S_fd.F0.γa.K2 - S.F.γa.K2) < 1e-3
    @test absmax(S_fd.F.γt.K2 + S_fd.F0.γt.K2 - S.F.γt.K2) < 1e-3
    @test absmax(S_fd.F.γp.K3 + S_fd.F0.γp.K3 - S.F.γp.K3) < 1e-3
    @test absmax(S_fd.F.γa.K3 + S_fd.F0.γa.K3 - S.F.γa.K3) < 1e-3
    @test absmax(S_fd.F.γt.K3 + S_fd.F0.γt.K3 - S.F.γt.K3) < 1e-3

    Ω = MatsubaraFrequency(T, 0, Boson)
    ν = MatsubaraFrequency(T, 1, Fermion)
    νp = MatsubaraFrequency(T, -2, Fermion)
    for ch in [:γa, :γp, :γt]
        z1 = getproperty(S_fd2.F, ch)(Ω, ν, νp) + getproperty(S_fd2.F0, ch)(Ω, ν, νp)
        z2 = getproperty(S.F, ch)(Ω, ν, νp)
        @test z1 ≈ z2 atol = 5e-4
    end
    for ch in [pCh, tCh, aCh], sp in [pSp, xSp]
        @test S_fd2.F(Ω, ν, νp, ch, sp) ≈ S.F(Ω, ν, νp, ch, sp) atol = 5e-4
    end

    # Test Vertex evaluation
    F = S_fd.F

    Ω = MatsubaraFrequency(T, 1, Boson)
    ω = MatsubaraFrequency(T, 3, Fermion)

    # If one of the fermionic frequency is νInf, other channels should not be evaluated
    for F0 in (true, false), ch in (pCh, tCh, aCh), sp in (pSp, xSp, dSp)
        x = F(Ω, ω, νInf, ch, sp; F0, γp = (ch === pCh), γt = (ch === tCh), γa = (ch === aCh))
        y = F(Ω, ω, νInf, ch, sp; F0)
        @test x ≈ y

        x = F(Ω, νInf, ω, ch, sp; F0, γp = (ch === pCh), γt = (ch === tCh), γa = (ch === aCh))
        y = F(Ω, νInf, ω, ch, sp; F0)
        @test x ≈ y
    end

    @test F(Ω, ω, νInf, pCh, pSp) ≈ F.γp(Ω, ω, νInf) + F.F0.γp(Ω, ω, νInf) + F.F0.F0(Ω, ω, νInf, pCh, pSp)
    @test F(Ω, ω, νInf, tCh, pSp) ≈ F.γt(Ω, ω, νInf) + F.F0.γt(Ω, ω, νInf) + F.F0.F0(Ω, ω, νInf, tCh, pSp)
    @test F(Ω, ω, νInf, aCh, pSp) ≈ F.γa(Ω, ω, νInf) + F.F0.γa(Ω, ω, νInf) + F.F0.F0(Ω, ω, νInf, aCh, pSp)
    @test F(Ω, νInf, ω, pCh, pSp) ≈ F.γp(Ω, νInf, ω) + F.F0.γp(Ω, νInf, ω) + F.F0.F0(Ω, νInf, ω, pCh, pSp)
    @test F(Ω, νInf, ω, tCh, pSp) ≈ F.γt(Ω, νInf, ω) + F.F0.γt(Ω, νInf, ω) + F.F0.F0(Ω, νInf, ω, tCh, pSp)
    @test F(Ω, νInf, ω, aCh, pSp) ≈ F.γa(Ω, νInf, ω) + F.F0.γa(Ω, νInf, ω) + F.F0.F0(Ω, νInf, ω, aCh, pSp)
end
