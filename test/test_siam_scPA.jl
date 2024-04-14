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
    nK1 = 4nmax
    nK2 = (nmax + 1, nmax)
    nK3 = (nmax + 1, nmax)

    for Q in [Float64, ComplexF64], mode in [:serial, :threads, :polyester, :hybrid]
        S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3, Q; e, T, D, Δ, U)
        init_sym_grp!(S)

        res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = mode);

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.05258541004593362, -0.03860700013157435, 0.03860700013157435, 0.05258541004593362]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.14065934981264078, 0.5551667095212016, 0.24352409131360997, 0.09849380953694975]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10926398023034017, -0.24371358931526044, -0.1603397125098747, -0.08160963485935688]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.01569768479115028, 0.15572656010297065, 0.04159218940186761, 0.008442087338796436]

        @test eltype(S.F) == Q
        @test eltype(S.Σ.data) == Q
    end
end


@testset "SIAM parquet doped" begin
    using MPI
    MPI.Init()

    T = 0.1
    U = 1.0
    e = 0.5
    Δ = π / 5
    D = 10.0

    nmax = 6
    nG  = 6nmax
    nK1 = 4nmax
    nK2 = (nmax + 1, nmax)
    nK3 = (nmax + 1, nmax)

    S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, D, Δ, U)
    init_sym_grp!(S)

    res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = :threads);

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.03903991568598151 - 0.16888644865715086im, -0.02532003648470073 - 0.17451691896616287im, 0.02532003648470073 - 0.17451691896616287im, 0.03903991568598151 - 0.16888644865715086im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.12758064200182956 - 3.796062856414873e-18im, 0.42916091514598903 - 1.2520488280775746e-8im, 0.21318205670662713 + 2.5040976561666116e-8im, 0.09029730232325764 + 1.1064286042753364e-8im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.13061534668813798 + 0.06501648894162436im, -0.24899004192424318 + 0.020742332961944773im, -0.18013175377843083 - 0.05338911104979782im, -0.10067321391574263 - 0.06605658951742963im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.012549142637799764 - 4.225677967262254e-18im, 0.09992985910164967 - 4.21724858022054e-8im, 0.03131559019227523 + 8.434497160215409e-8im, 0.006920207613575011 + 4.8593739940477936e-9im]
end
