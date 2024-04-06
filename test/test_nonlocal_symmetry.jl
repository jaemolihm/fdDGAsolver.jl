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
        Symmetry{2}(w -> sΣ_conj(w,)),
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


@testset "nonlocal symmetry vertex" begin
    T = 0.2
    U = 2.0
    μ = -2.0
    t1 = 1.0

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    nmax = 4
    nG  = 8nmax
    nΣ  = 8nmax
    nK1 = 4nmax
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    mK_G = BrillouinZoneMesh(BrillouinZone(8, k1, k2))
    mK_Γ = BrillouinZoneMesh(BrillouinZone(4, k1, k2))

    S = parquet_solver_hubbard_parquet_approximation(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1)

    fdDGAsolver.Dyson!(S)
    fdDGAsolver.bubbles!(S)

    # Set G0 to some auxiliary value for testing
    S.G0 = hubbard_bare_Green(meshes(S.G0)...; μ = 1.0, t1 = 0.5)
    fdDGAsolver.bubbles!(S.Π0pp, S.Π0ph, S.G0)

    # Compute BSE without symmetries
    fdDGAsolver.BSE_K1!(S, pCh);
    fdDGAsolver.BSE_K1!(S, aCh);
    fdDGAsolver.BSE_K1!(S, tCh);

    set!(S.F, S.Fbuff)

    fdDGAsolver.BSE_K1!(S, pCh);
    fdDGAsolver.BSE_K1!(S, aCh);
    fdDGAsolver.BSE_K1!(S, tCh);

    fdDGAsolver.BSE_L_K2!(S, pCh);
    fdDGAsolver.BSE_L_K2!(S, aCh);
    fdDGAsolver.BSE_L_K2!(S, tCh);
    fdDGAsolver.BSE_K2!(S, pCh);
    fdDGAsolver.BSE_K2!(S, aCh);
    fdDGAsolver.BSE_K2!(S, tCh);

    Q = eltype(S.F)

    # build vertices
    g = MatsubaraMesh(temperature(S), 2 * numK1(S), Fermion)
    Γpx = MeshFunction(meshes(S.F.γp.K3, 1), g, meshes(S.F.γp.K3, 3), meshes(S.F.γp.K3, 4); data_t=Q)
    F0p = copy(Γpx)
    F0a = copy(Γpx)
    F0t = copy(Γpx)

    Threads.@threads for i in CartesianIndices(Γpx.data)
        Ω  = value(meshes(Γpx, 1)[i.I[1]])
        ν  = value(meshes(Γpx, 2)[i.I[2]])
        νp = value(meshes(Γpx, 3)[i.I[3]])
        P  = value(meshes(Γpx, 4)[i.I[4]])
        Γpx[i] = S.F( Ω, ν, νp, P, kSW, kSW, pCh, xSp; F0=false, γp=false)
        F0p[i] = S.F0(Ω, ν, νp, P, kSW, kSW, pCh, xSp)
        F0a[i] = S.F0(Ω, ν, νp, P, kSW, kSW, aCh, pSp)
        F0t[i] = S.F0(Ω, ν, νp, P, kSW, kSW, tCh, pSp) * 2 - F0a[i]
    end

    Γpp = MeshFunction(meshes(S.F.γp.K3, 1), meshes(S.F.γp.K3, 2), g, meshes(S.F.γp.K3, 4); data_t=Q)
    Γa = deepcopy(Γpp)
    Γt = deepcopy(Γpp)
    Fp = deepcopy(Γpp)
    Fa = deepcopy(Γpp)
    Ft = deepcopy(Γpp)

    Threads.@threads for i in CartesianIndices(Γpp.data)
        Ω  = value(meshes(Γpp, 1)[i.I[1]])
        ν  = value(meshes(Γpp, 2)[i.I[2]])
        νp = value(meshes(Γpp, 3)[i.I[3]])
        P  = value(meshes(Γpx, 4)[i.I[4]])
        Γpp[i] = S.F(Ω, ν, νp, P, kSW, kSW, pCh, pSp; F0=false, γp=false)
        Γa[i]  = S.F(Ω, ν, νp, P, kSW, kSW, aCh, pSp; F0=false, γa=false)
        Γt[i]  = S.F(Ω, ν, νp, P, kSW, kSW, tCh, pSp; F0=false, γt=false) * 2 - Γa[i]
        Fp[i]  = S.F(Ω, ν, νp, P, kSW, kSW, pCh, pSp)
        Fa[i]  = S.F(Ω, ν, νp, P, kSW, kSW, aCh, pSp)
        Ft[i]  = S.F(Ω, ν, νp, P, kSW, kSW, tCh, pSp) * 2 - Fa[i]
    end

    fdDGAsolver.BSE_L_K3!(S, Γpp, F0p, pCh)
    fdDGAsolver.BSE_L_K3!(S, Γa, F0a, aCh)
    fdDGAsolver.BSE_L_K3!(S, Γt, F0t, tCh)

    fdDGAsolver.BSE_K3!(S, Γpx, Fp, F0p, pCh)
    fdDGAsolver.BSE_K3!(S, Γa, Fa, F0a, aCh)
    fdDGAsolver.BSE_K3!(S, Γt, Ft, F0t, tCh)

    # Check vertices are nonzero

    @test absmax(S.FL.γp.K2) > 1e-3
    @test absmax(S.FL.γa.K2) > 1e-3
    @test absmax(S.FL.γt.K2) > 1e-3
    @test absmax(S.FL.γp.K3) > 1e-3
    @test absmax(S.FL.γa.K3) > 1e-3
    @test absmax(S.FL.γt.K3) > 1e-3

    @test absmax(S.Fbuff.γp.K1) > 1e-3
    @test absmax(S.Fbuff.γa.K1) > 1e-3
    @test absmax(S.Fbuff.γt.K1) > 1e-3
    @test absmax(S.Fbuff.γp.K2) > 1e-3
    @test absmax(S.Fbuff.γa.K2) > 1e-3
    @test absmax(S.Fbuff.γt.K2) > 1e-3
    @test absmax(S.Fbuff.γp.K3) > 1e-3
    @test absmax(S.Fbuff.γa.K3) > 1e-3
    @test absmax(S.Fbuff.γt.K3) > 1e-3

    # Now initialize symmetries and test symmetry of the vertices
    fdDGAsolver.init_sym_grp!(S)

    @test S.SGppL[2](S.FL.γp.K2) < 1e-10
    @test S.SGphL[2](S.FL.γa.K2) < 1e-10
    @test S.SGphL[2](S.FL.γt.K2) < 1e-10

    @test S.SGppL[3](S.FL.γp.K3) < 1e-10
    @test S.SGphL[3](S.FL.γa.K3) < 1e-10
    @test S.SGphL[3](S.FL.γt.K3) < 1e-10

    @test S.SGpp[1](S.Fbuff.γp.K1) < 1e-10
    @test S.SGph[1](S.Fbuff.γa.K1) < 1e-10
    @test S.SGph[1](S.Fbuff.γt.K1) < 1e-10

    @test S.SGpp[2](S.Fbuff.γp.K2) < 6e-3  # This improves for large nmax
    @test S.SGph[2](S.Fbuff.γa.K2) < 6e-3  # This improves for large nmax
    @test S.SGph[2](S.Fbuff.γt.K2) < 6e-3  # This improves for large nmax

    @test S.SGpp[3](S.Fbuff.γp.K3) < 4e-3   # This improves for large nmax
    @test S.SGph[3](S.Fbuff.γa.K3) < 4e-3   # This improves for large nmax
    @test S.SGph[3](S.Fbuff.γt.K3) < 4e-3   # This improves for large nmax

end
