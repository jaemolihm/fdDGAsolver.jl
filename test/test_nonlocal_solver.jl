using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using Test

@testset "NL_ParquetSolver" begin
    using MPI
    MPI.Init()

    T = 0.5
    U = 1.0
    μ = 0.
    t1 = 1.0

    nmax = 4
    nG  = 8nmax
    nΣ  = 8nmax
    nK1 = 4nmax
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK_G = BrillouinZoneMesh(BrillouinZone(8, k1, k2))
    mK_Γ = BrillouinZoneMesh(BrillouinZone(4, k1, k2))

    S = parquet_solver_hubbard_parquet_approximation(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1)

    @test fdDGAsolver.numG(S) == nG
    @test fdDGAsolver.numΣ(S) == nΣ
    @test fdDGAsolver.numK1(S) == nK1
    @test fdDGAsolver.numK2(S) == nK2
    @test fdDGAsolver.numK3(S) == nK3
    @test fdDGAsolver.numP_G(S) == length(mK_G)
    @test fdDGAsolver.numP_Γ(S) == length(mK_Γ)

    # Test flatten and unflatten

    S_copy = parquet_solver_hubbard_parquet_approximation(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1)
    set!(S_copy.F, 0)
    S.F.γa.K3.data .= rand(size(S.F.γa.K3.data)...)
    unflatten!(S_copy, flatten(S))
    @test absmax(S.F.γa.K3 - S_copy.F.γa.K3) < 1e-10
end
