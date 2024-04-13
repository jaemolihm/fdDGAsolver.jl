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
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    for Q in [Float64, ComplexF64], mode in [:serial, :threads, :polyester, :hybrid]
        S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3, Q; e, T, D, Δ, U)
        init_sym_grp!(S)

        res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = mode);

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.05238088346448162, -0.03852919004835304, 0.03852919004835304, 0.05238088346448162]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.14088981650230273, 0.5557618273628016, 0.2438822514592749, 0.09864807393697142]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10914131687049806, -0.24358383229751174, -0.16020179899085216, -0.08151156198526349]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.015874249815902318, 0.15608899753264494, 0.04184022623421136, 0.008568255975853948]

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
    nK2 = (2nmax, nmax)
    nK3 = (2nmax, nmax)

    S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, D, Δ, U)
    init_sym_grp!(S)

    res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = :threads);

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.03893718775529078 - 0.1686604375262501im, -0.025289432965365875 - 0.17438194103733454im, 0.025289432965365875 - 0.17438194103733454im, 0.03893718775529078 - 0.1686604375262501im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.12767301017646945 - 2.019326546254252e-18im, 0.42928070843884575 + 8.049669336139359e-8im, 0.21330877520460267 - 1.609933867148831e-7im, 0.09035489003605932 + 1.668113572112785e-8im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.13054376394836842 + 0.06501573275114048im, -0.24889673612727514 + 0.02074647907116367im, -0.1800456441256735 - 0.05339726481061868im, -0.10062595807812454 - 0.06604952267404215im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.012624888641187056 - 4.230421351766877e-18im, 0.10002634714060862 - 4.1044381634941204e-7im, 0.0314131620948432 + 8.208876327070597e-7im, 0.006971806554142257 - 7.38886040075296e-9im]

end
