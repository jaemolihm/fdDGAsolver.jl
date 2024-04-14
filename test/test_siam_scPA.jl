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

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.052425361159270094, -0.03846395893725492, 0.03846395893725492, 0.052425361159270094]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.1404675439940127, 0.554775885680526, 0.2432583675626251, 0.09834804227300667]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10940265466750497, -0.24392921295237002, -0.1605149546054454, -0.08172273147933724]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.015532444663253868, 0.15542333636407796, 0.041371706478589815, 0.008312655396834717]

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
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, D, Δ, U)
    init_sym_grp!(S)

    res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = :threads);

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.039054010351296534 - 0.16892145043571732im, -0.025323879793382974 - 0.17455375938303208im, 0.025323879793382974 - 0.17455375938303208im, 0.039054010351296534 - 0.16892145043571732im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.12750715961765344 - 1.5341460740669888e-18im, 0.429013001325934 + 3.185430044963427e-7im, 0.2130721066392593 - 6.370860090029709e-7im, 0.09024952100792166 + 2.19792403668919e-8im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.13070822716996025 + 0.06505544807689725im, -0.24911926257775818 + 0.020751026653492682im, -0.18024582717683763 - 0.053413631806220295im, -0.10074472730492225 - 0.06610529093432214im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.012494478685123107 - 1.6967763999398147e-18im, 0.09984276213809809 - 3.7678770883081197e-7im, 0.031241468280110197 + 7.535754176568093e-7im, 0.006881882932420506 - 1.8430189068804524e-8im]

end
