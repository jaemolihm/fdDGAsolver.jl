using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using Test

@testset "nonlocal symmetry G" begin
    # Test symmetry of the Green function
    using fdDGAsolver: sΣ_conj, sΣ_ref, sΣ_rot

    using MPI
    MPI.Init()

    T = 0.5

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    mG  = MatsubaraMesh(T, 5, Fermion)
    mK = BrillouinZoneMesh(BrillouinZone(4, k1, k2))

    G = hubbard_bare_Green(mG, mK; μ = 0.3, t1 = 1.0, t2 = 0.2, t3 = -0.4)

    # compute the symmetry group
    SG = SymmetryGroup(Symmetry{2}[
        Symmetry{2}(w -> sΣ_conj(w, mK)),
        Symmetry{2}(w -> sΣ_ref(w, mK)),
        Symmetry{2}(w -> sΣ_rot(w, mK))
    ], G);

    # Check G satisfies symmetry
    @test SG(G) < 1e-14

    # symmetrize G_sym and compare to G
    G_sym = MeshFunction(mG, mK)
    for class in SG.classes
        G_sym[class[1][1]] = G[class[1][1]]
    end
    SG(G_sym)
    @test G_sym == G

    # symmetrize G_sym and compare to G using InitFunction
    InitFunc = InitFunction{2, ComplexF64}(w -> G[w...])
    for mode in [:serial, :threads, :polyester, :hybrid]
        set!(G_sym, 0)
        SG(G_sym, InitFunc; mode = :serial)
        @test G_sym == G
    end
end;

@testset "nonlocal symmetry vertex scPA" begin
    T = 0.5
    U = 2.0
    μ = -2.0
    t1 = 1.0
    t2 = -0.5

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    nmax = 3
    nG  = 8nmax
    nK1 = 4nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    mK_G = BrillouinZoneMesh(BrillouinZone(6, k1, k2))
    mK_Γ = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    S = parquet_solver_hubbard_parquet_approximation(nG, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1, t2, mode = :threads)

    # Solve without symmetries

    res = fdDGAsolver.solve!(S; strategy = :scPA, update_Σ = false, verbose = false);

    # Check vertices are nonzero

    @testset "value" begin
        @test absmax(S.F.γp.K1) > 1e-3
        @test absmax(S.F.γa.K1) > 1e-3
        @test absmax(S.F.γt.K1) > 1e-3
        @test absmax(S.F.γp.K2) > 1e-3
        @test absmax(S.F.γa.K2) > 1e-3
        @test absmax(S.F.γt.K2) > 1e-3
        @test absmax(S.F.γp.K3) > 1e-3
        @test absmax(S.F.γa.K3) > 1e-3
        @test absmax(S.F.γt.K3) > 1e-3
    end

    # Now initialize symmetries and test symmetry of the vertices

    init_sym_grp!(S)

    @testset "symmetry error" begin
        @test S.SGpp[1](S.F.γp.K1) < 1e-10
        @test S.SGph[1](S.F.γa.K1) < 1e-10
        @test S.SGph[1](S.F.γt.K1) < 1e-10

        # These error can be reduced using larger nmax
        @test S.SGpp[2](S.F.γp.K2) < 1e-3
        @test S.SGph[2](S.F.γa.K2) < 1e-3
        @test S.SGph[2](S.F.γt.K2) < 2e-3

        @test S.SGpp[3](S.F.γp.K3) < 2e-2
        @test S.SGph[3](S.F.γa.K3) < 1e-3
        @test S.SGph[3](S.F.γt.K3) < 2e-3
    end

end


@testset "nonlocal symmetry vertex fdPA" begin
    T = 0.5
    U = 2.0
    μ = -2.0
    t1 = 1.0

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    nmax = 4
    nG  = 8nmax
    nK1 = 4nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    mK_G = BrillouinZoneMesh(BrillouinZone(6, k1, k2))
    mK_Γ = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    S = parquet_solver_hubbard_parquet_approximation(nG, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1, mode = :polyester)

    # Set G0 to some auxiliary value for testing
    S.G0 = hubbard_bare_Green(meshes(S.G0)...; μ = 1.0, t1 = 0.5)
    fdDGAsolver.bubbles_momentum_space!(S.Π0pp, S.Π0ph, S.G0)

    # Solve without symmetries

    res = fdDGAsolver.solve!(S; strategy = :fdPA, update_Σ = false, verbose = false);

    # Check vertices are nonzero

    @testset "value" begin
        @test absmax(S.FL.γp.K2) > 1e-4
        @test absmax(S.FL.γa.K2) > 1e-4
        @test absmax(S.FL.γt.K2) > 1e-4
        @test absmax(S.FL.γp.K3) > 1e-4
        @test absmax(S.FL.γa.K3) > 1e-4
        @test absmax(S.FL.γt.K3) > 1e-4
    end

    # Now initialize symmetries and test symmetry of the vertices

    init_sym_grp!(S)

    @testset "symmetry error" begin
        # These symmetry are exact only when the fdPA is converged (?)
        @test S.SGppL[2](S.FL.γp.K2) < 3e-3
        @test S.SGphL[2](S.FL.γa.K2) < 3e-4
        @test S.SGphL[2](S.FL.γt.K2) < 1e-3
        @test S.SGppL[3](S.FL.γp.K3) < 3e-3
        @test S.SGphL[3](S.FL.γa.K3) < 4e-4
        @test S.SGphL[3](S.FL.γt.K3) < 1e-3
    end

end
