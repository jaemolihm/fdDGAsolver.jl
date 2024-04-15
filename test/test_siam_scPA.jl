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

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.0523590263383733, -0.03853433592128489, 0.03853433592128489, 0.0523590263383733]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.14055317640852846, 0.5554831858336376, 0.24351787621784904, 0.0983303256206339]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10922041766435386, -0.24372921625119187, -0.16033570312419304, -0.08151737657250269]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.01566536608641031, 0.155875834817708, 0.04158811140125055, 0.008407750947301916]

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

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.03902675700461936 - 0.16873041234414765im, -0.02533927703077011 - 0.1745017903854828im, 0.02533927703077011 - 0.1745017903854828im, 0.03902675700461936 - 0.16873041234414765im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.12744069212778786 + 3.4035190557859594e-5im, 0.4291657543713851 + 2.071753451861202e-5im, 0.2130766562741666 - 4.766687369078173e-5im, 0.0901357758971778 - 9.770482285323927e-6im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.13056824833295191 + 0.06498315681982186im, -0.24897605140900786 + 0.020731512652888986im, -0.18010276958744215 - 0.05336136735812157im, -0.10059992564758968 - 0.06601816016380756im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.01249346487390409 + 6.393014824171219e-5im, 0.09992820871201978 + 2.0722293662767286e-5im, 0.031265287796739934 - 5.315012470571013e-5im, 0.006867271042880291 - 6.301543569672634e-5im]
end
