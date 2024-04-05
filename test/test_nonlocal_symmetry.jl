using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using Test

@testset "nonlocal symmetry G" begin
    # Test symmetry of the Green function
    using fdDGAsolver: sΣ_conj, sΣ_ref, sΣ_rot

    T = 0.5

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    mG  = MatsubaraMesh(T, 5, Fermion)
    mK = BrillouinZoneMesh(BrillouinZone(4, k1, k2))

    G = hubbard_bare_Green(mG, mK; μ = 0.3, t1 = 1.0, t2 = 0.2, t3 = -0.4)

    # compute the symmetry group
    SG = SymmetryGroup(Symmetry{2}[
        Symmetry{2}(w -> sΣ_conj(w, mK)),
        Symmetry{2}(w -> sΣ_ref( w, mK)),
        Symmetry{2}(w -> sΣ_rot( w, mK))
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
