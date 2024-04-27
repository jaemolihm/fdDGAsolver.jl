using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using HDF5
using Test

@testset "MBEVertex" begin
    using MPI
    MPI.Init()
    using fdDGAsolver: MBEVertex, asymptotic_to_mbe, mbe_to_asymptotic

    T = 0.5
    U = 2.0
    numK1 = 10
    numK2 = (7, 7)
    numK3 = (5, 5)

    Ω = MatsubaraFrequency(T, 0, Boson)
    ν = MatsubaraFrequency(T, 0, Fermion)
    ω = MatsubaraFrequency(T, 1, Fermion)

    Γ = MBEVertex(RefVertex(T, U), T, numK1, numK2, numK3)

    # Test bare vertex
    unflatten!(Γ, zeros(eltype(Γ), length(Γ)))
    for Ch in [aCh, pCh, tCh], (Sp, U_Sp) in zip([pSp, dSp, xSp], [U, U, -U])
        @test Γ(Ω, ν, ω, Ch, Sp) ≈ U_Sp
    end

    # Test InfiniteMatsubaraFrequency evaluation
    unflatten!(Γ, rand(eltype(Γ), length(Γ)))
    for (Ch, γ) in zip([aCh, pCh, tCh], [Γ.γa, Γ.γp, Γ.γt])
        @test Γ(Ω,    ν, νInf, Ch, pSp) ≈ U + γ.K1(Ω) + γ.K2(Ω, ν)
        @test Γ(Ω, νInf,    ω, Ch, pSp) ≈ U + γ.K1(Ω) + γ.K2(Ω, ω)
        @test Γ(Ω, νInf, νInf, Ch, pSp) ≈ U + γ.K1(Ω)
    end

    # Test copy
    Γ_copy = copy(Γ)
    @test Γ == Γ_copy
    @test Γ.F0 !== Γ_copy.F0
    @test Γ.γp !== Γ_copy.γp

    # Test flatten and unflatten
    set!(Γ_copy, 0)
    @test all(flatten(Γ_copy) .== 0)
    @test Γ_copy != Γ

    unflatten!(Γ_copy, flatten(Γ))
    @test Γ_copy == Γ

    # Test IO
    testfile = dirname(@__FILE__) * "/test.h5"
    file = h5open(testfile, "w")
    save!(file, "f", Γ)
    close(file)

    file = h5open(testfile, "r")
    Γ_read = fdDGAsolver.load_vertex(MBEVertex, file, "f")
    @test Γ_read == Γ
    @test Γ_read isa MBEVertex
    close(file)

    rm(testfile; force=true)


    # Test conversion between asymptotic and MBE vertices
    F = Vertex(RefVertex(T, U), T, numK1, numK2, numK3)
    unflatten!(F, rand(ComplexF64, length(F)))

    F_mbe = asymptotic_to_mbe(F)
    Ω = MatsubaraFrequency(T, -1, Boson)
    ν = MatsubaraFrequency(T, -1, Fermion)
    ω = MatsubaraFrequency(T, 1, Fermion)
    for Ch in (aCh, pCh, tCh), Sp in (pSp, dSp, xSp)
        @test F_mbe(Ω, ν, ω, Ch, Sp) ≈ F(Ω, ν, ω, Ch, Sp)
    end

    F_new = mbe_to_asymptotic(F_mbe)
    for Ch in (aCh, pCh, tCh), Sp in (pSp, dSp, xSp)
        @test F_new(Ω, ν, ω, Ch, Sp) ≈ F_mbe(Ω, ν, ω, Ch, Sp)
    end
    @test flatten(F) ≈ flatten(F_new)
end


@testset "SIAM parquet MBE" begin
    using MPI
    MPI.Init()

    T = 0.1
    U = 1.0
    e = 0.5
    Δ = π / 5
    D = 10.0

    nmax = 20
    nG  = 6nmax
    nK1 = 4nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    S1 = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, D, Δ, U, VT = MBEVertex)
    init_sym_grp!(S1)
    fdDGAsolver.solve!(S1; strategy = :scPA, tol = 1e-5, verbose = false);

    S2 = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, D, Δ, U)
    init_sym_grp!(S2)
    fdDGAsolver.solve!(S2; strategy = :scPA, tol = 1e-5, verbose = false);

    @test absmax(S1.Σ - S2.Σ) < 2e-6

    @test absmax(S1.F.γa.K1 - S2.F.γa.K1) < 2e-5
    @test absmax(S1.F.γp.K1 - S2.F.γp.K1) < 2e-5
    @test absmax(S1.F.γt.K1 - S2.F.γt.K1) < 2e-5
    @test absmax(S1.F.γa.K2 - S2.F.γa.K2) < 2e-5
    @test absmax(S1.F.γp.K2 - S2.F.γp.K2) < 2e-5
    @test absmax(S1.F.γt.K2 - S2.F.γt.K2) < 2e-5

    Ω = MatsubaraFrequency(T, 0, Boson)
    νs = MatsubaraMesh(T, 20, Fermion)

    for Ch in [aCh, pCh, tCh], Sp in [pSp, dSp, xSp]
        z1 = [S1.F(Ω, ν, ω, Ch, Sp) for ν in value.(νs), ω in value.(νs)]
        z2 = [S2.F(Ω, ν, ω, Ch, Sp) for ν in value.(νs), ω in value.(νs)]
        @test maximum(abs.(z1 .- z2)) < 1e-4
    end
end


@testset "SIAM fdPA MBE" begin
    using MPI
    MPI.Init()
    using fdDGAsolver: MBEVertex

    T = 0.1
    U = 1.0
    D = 10.0
    e = -0.3
    Δ = π / 3

    nmax = 12
    nG  = 8nmax
    nK1 = 8nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    VT = MBEVertex
    # VT = Vertex
    update_Σ = true
    # update_Σ = false

    # scPA for the reference point
    S0 = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, U, Δ, D, VT)
    init_sym_grp!(S0)
    res = fdDGAsolver.solve!(S0; strategy = :scPA, parallel_mode = :threads, verbose = false, update_Σ);


    # fdPA should run and converge to the same point (trivial case of G0 = Σ0 = Π0 = 0)
    S0_fd = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, U, Δ, D, VT)
    init_sym_grp!(S0_fd)
    res = fdDGAsolver.solve!(S0_fd; strategy = :fdPA, parallel_mode = :threads, verbose = false, update_Σ);
    @test absmax(S0.Σ - S0_fd.Σ) < 1e-10
    @test maximum(abs.(flatten(S0.F) .- flatten(S0_fd.F))) < 1e-10


    # scPA for the target point
    Δ_fd = π / 5
    e_fd = 0.5
    D_fd = 20.0
    S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e = e_fd, T, U, Δ = Δ_fd, D = D_fd, VT)
    init_sym_grp!(S)
    res = fdDGAsolver.solve!(S; strategy = :scPA, parallel_mode = :threads, verbose = false, update_Σ);

    # fdPA from the reference to the target
    Gbare = fdDGAsolver.siam_bare_Green(meshes(S0.G, Val(1)); e = e_fd, Δ = Δ_fd, D = D_fd)

    S_fd = ParquetSolver(nK1, nK2, nK3, Gbare, S0.G, S0.Σ, S0.F, VT,)
    init_sym_grp!(S_fd)
    res = fdDGAsolver.solve!(S_fd; strategy = :fdPA, parallel_mode = :threads, verbose = false, update_Σ);

    @test absmax(S_fd.Σ - S.Σ) < 1e-5

    @test absmax(S_fd.F.γp.K1 + S_fd.F0.γp.K1 - S.F.γp.K1) < 2e-4
    @test absmax(S_fd.F.γa.K1 + S_fd.F0.γa.K1 - S.F.γa.K1) < 2e-4
    @test absmax(S_fd.F.γt.K1 + S_fd.F0.γt.K1 - S.F.γt.K1) < 2e-4
    @test absmax(S_fd.F.γp.K2 + S_fd.F0.γp.K2 - S.F.γp.K2) < 1e-4
    @test absmax(S_fd.F.γa.K2 + S_fd.F0.γa.K2 - S.F.γa.K2) < 1e-4
    @test absmax(S_fd.F.γt.K2 + S_fd.F0.γt.K2 - S.F.γt.K2) < 1e-4
    @test absmax(S_fd.F.γp.K3 + S_fd.F0.γp.K3 - S.F.γp.K3) < 5e-4
    @test absmax(S_fd.F.γa.K3 + S_fd.F0.γa.K3 - S.F.γa.K3) < 5e-4
    @test absmax(S_fd.F.γt.K3 + S_fd.F0.γt.K3 - S.F.γt.K3) < 5e-4


    # fdPA with different box sizes for the vertex
    nmax = 9
    nK1 = 8nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    Gbare = fdDGAsolver.siam_bare_Green(meshes(S0.G, Val(1)); e = e_fd, Δ = Δ_fd, D = D_fd)

    S_fd2 = ParquetSolver(nK1, nK2, nK3, Gbare, S0.G, S0.Σ, S0.F, VT)
    init_sym_grp!(S_fd2)
    res = fdDGAsolver.solve!(S_fd2; strategy = :fdPA, parallel_mode = :threads, verbose = false, update_Σ);

    @test absmax(S_fd2.Σ - S.Σ) < 2e-3

    Ω = MatsubaraFrequency(T, 0, Boson)
    ν = MatsubaraFrequency(T, 1, Fermion)
    νp = MatsubaraFrequency(T, -2, Fermion)
    for ch in [:γa, :γp, :γt]
        z1 = getproperty(S_fd2.F, ch)(Ω, ν, νp) + getproperty(S_fd2.F0, ch)(Ω, ν, νp)
        z2 = getproperty(S.F, ch)(Ω, ν, νp)
        @test z1 ≈ z2 atol = 7e-4
    end
    for ch in [pCh, tCh, aCh], sp in [pSp, xSp]
        @test S_fd2.F(Ω, ν, νp, ch, sp) ≈ S.F(Ω, ν, νp, ch, sp) atol = 6e-4
    end
end
