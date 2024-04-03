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
    nΣ  = 6nmax
    nK1 = 4nmax
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    S = parquet_solver_siam_parquet_approximation(nG, nΣ, nK1, nK2, nK3; e, T, D, Δ, U)
    fdDGAsolver.init_sym_grp!(S)

    res = fdDGAsolver.solve!(S; strat = :scPA, verbose = false, parallel_mode = :threads);

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.03824070097780221 - 0.1680223364784656im, -0.025009124539466316 - 0.1737636175072827im, 0.025009124539466316 - 0.1737636175072827im, 0.03824070097780221 - 0.1680223364784656im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.12769493880529753 + 4.6654794404848284e-5im, 0.42934963842995794 + 2.086772186614021e-5im, 0.2134122166709921 - 5.027788348710732e-5im, 0.09031817322744154 - 2.336206882038395e-5im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.13065147595233986 + 0.06509002427844468im, -0.24900141511816531 + 0.020761402759562602im, -0.18018135126866128 - 0.05344071488151015im, -0.1006825715022467 - 0.06612971748206087im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.01261887626048349 + 8.55803849670883e-5im, 0.09998789236029004 + 2.85065903711996e-5im, 0.031419846299068126 - 7.268285182552409e-5im, 0.006958319537886918 - 8.630020460693878e-5im]

end
