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

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.052677436018339505, -0.03862941505634685, 0.03862941505634685, 0.052677436018339505]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.14084395171160435, 0.5555241595664626, 0.2437814443626537, 0.09862826154481966]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10911481448900334, -0.24353982687539988, -0.16016059391499146, -0.08149900918988015]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.015864568611300506, 0.15599216634553142, 0.04181042522383113, 0.00856462617746974]

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

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.039026473801688645 - 0.16886683628896376im, -0.025329905006231813 - 0.1745387085082399im, 0.025329905006231813 - 0.1745387085082399im, 0.039026473801688645 - 0.16886683628896376im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.12768953132137426 - 3.950561665994057e-18im, 0.4293556134899288 + 3.4632855614576914e-7im, 0.21333028991800573 - 6.926571122938822e-7im, 0.09037035942779767 + 2.0250214332249807e-8im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.1305168913013253 + 0.06499410707330057im, -0.24888276715347427 + 0.02074062163336891im, -0.18001962216711134 - 0.053381590298030204im, -0.10060011389259807 - 0.06602349773999593im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.012629222694495082 - 3.939719644269202e-18im, 0.10005926484157159 - 3.9779376121254545e-7im, 0.03142066846856429 + 7.955875224239688e-7im, 0.006974695160991964 - 2.2327533921553258e-8im]

end
