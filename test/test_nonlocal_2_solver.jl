using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using Test

@testset "NL2_ParquetSolver" begin
    using MPI
    MPI.Init()

    T = 0.5
    U = 3.0
    μ = 0.
    t1 = 1.0

    nmax = 4
    nG  = 4nmax
    nK1 = 3nmax
    nK2 = (nmax + 1, nmax)
    nK3 = (nmax + 1, nmax)

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK_G = BrillouinZoneMesh(BrillouinZone(8, k1, k2))
    mK_Γ = BrillouinZoneMesh(BrillouinZone(4, k1, k2))

    S = parquet_solver_hubbard_parquet_approximation_NL2(nG, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1)

    # Fill in random dummy data
    unflatten!(S, rand(ComplexF64, length(flatten(S))))

    @test fdDGAsolver.numG(S) == nG
    @test fdDGAsolver.numΣ(S) == nG
    @test fdDGAsolver.numK1(S) == nK1
    @test fdDGAsolver.numK2(S) == nK2
    @test fdDGAsolver.numK3(S) == nK3
    @test fdDGAsolver.numP_G(S) == length(mK_G)
    @test fdDGAsolver.numP_Γ(S) == length(mK_Γ)

    # Test flatten and unflatten

    S_copy = parquet_solver_hubbard_parquet_approximation_NL2(nG, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1)
    set!(S_copy.F, 0)
    unflatten!(S_copy, flatten(S))
    @test absmax(S.F.γa.K3 - S_copy.F.γa.K3) < 1e-10

    # Test IO

    testfile = dirname(@__FILE__) * "/test.h5"
    f = h5open(testfile, "w")
    save!(f, "", S)
    close(f)

    f = h5open(testfile, "r")
    F_load = fdDGAsolver.load_vertex(NL2_Vertex, f, "F")
    Σ_load = load_mesh_function(f, "Σ")
    close(f)

    rm(testfile; force=true)

    @test F_load isa NL2_Vertex
    @test F_load == S.F
    @test Σ_load == S.Σ
end
