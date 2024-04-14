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

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.05258173635464072, -0.0386033198206943, 0.0386033198206943, 0.05258173635464072]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.14065614353959735, 0.5551492604569783, 0.24352398698727928, 0.0984915204153704]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10926687221731941, -0.2437171933054847, -0.16034446386486742, -0.08161197678730553]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.01569463566113899, 0.15571603357574673, 0.04158976156120589, 0.008439771814032436]

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

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.03904045468186027 - 0.16888608711235464im, -0.025320695656877012 - 0.1745171029147528im, 0.025320695656877012 - 0.1745171029147528im, 0.03904045468186027 - 0.16888608711235464im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.1275801783904821 + 1.4040418133687284e-18im, 0.4291462955108083 + 2.94926575256862e-7im, 0.21317870936965377 - 5.898531504930712e-7im, 0.09029701372175188 + 2.6968243935458087e-8im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.1306184842668937 + 0.06501546768850777im, -0.24899350313946983 + 0.02074132335596881im, -0.1801356076265181 - 0.0533869048475626im, -0.10067557334944115 - 0.06605611794391823im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.01254659576252539 + 2.1230033789981784e-18im, 0.09992449679574376 - 3.902185727416043e-7im, 0.031312082694024586 + 7.804371454964741e-7im, 0.006918306654525544 - 2.1261440404345996e-8im]
end
