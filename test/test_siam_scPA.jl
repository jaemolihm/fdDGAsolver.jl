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
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    for Q in [Float64, ComplexF64], mode in [:serial, :threads, :polyester, :hybrid]
        S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3, Q; e, T, D, Δ, U)
        init_sym_grp!(S)

        res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = mode);

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.052138235296134906, -0.03838544776344314, 0.03838544776344314, 0.052138235296134906]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.13203850929270397, 0.5403615530152339, 0.2333246221064017, 0.09056300899983459]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10420799999591804, -0.2403951910434166, -0.15592452265704748, -0.07622568434721624]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.013898648018808482, 0.1499562726081748, 0.03867632419082161, 0.007160400841240708]

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

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.0389123277075552 - 0.16855090184215607im, -0.025252640312580586 - 0.17429637478745583im, 0.025252640312580586 - 0.17429637478745583im, 0.0389123277075552 - 0.16855090184215607im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.11925962005661812 + 8.57514999021054e-5im, 0.416232811242488 + 3.319936929625957e-5im, 0.20353141073439696 - 8.209974062547027e-5im, 0.08259294412067451 - 7.660306021755952e-5im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.12570450372739117 + 0.06583917638195431im, -0.24548654160724023 + 0.021014183409764874im, -0.17578586892780296 - 0.05408344507941078im, -0.09544981624337806 - 0.06686768343644132im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.011016969129875598 + 7.455601769977568e-5im, 0.09533547032272821 + 2.4477973720973318e-5im, 0.028843799251846686 - 6.260706942592786e-5im, 0.005828299701446311 - 7.32415771550901e-5im]
end
