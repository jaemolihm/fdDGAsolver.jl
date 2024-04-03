using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Test

@testset "SIAM parquet half-filling" begin
    using MPI
    MPI.Init()

    T = 0.1
    U = 1.0
    e = 0.0
    Δ = π / 5
    D = 10.0

    nmax = 6
    nG  = 6nmax
    nΣ  = 6nmax
    nK1 = 4nmax
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    for Q in [Float64, ComplexF64], mode in [:serial, :threads, :polyester, :hybrid]
        S = parquet_solver_siam_parquet_approximation(nG, nΣ, nK1, nK2, nK3, Q; e, T, D, Δ, U)
        fdDGAsolver.init_sym_grp!(S)

        res = fdDGAsolver.solve!(S; strat = :scPA, verbose = false, parallel_mode = mode);

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.05172702999366102, -0.03827565862415375, 0.03827565862415375, 0.05172702999366102]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.1410031933986213, 0.5564948604568605, 0.24418625119203477, 0.09866033387273063]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10914613070142642, -0.24367413144140015, -0.16026345738625844, -0.08145812439167487]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.015942280583863207, 0.15636082345188412, 0.04195891486488509, 0.0086191401115509]

        @test eltype(S.F) == Q
        @test eltype(S.Σ.data) == Q
    end
end
