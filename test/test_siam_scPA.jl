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

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.05239493482994168, -0.03850591686283782, 0.03850591686283782, 0.05239493482994168]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.14081343169023602, 0.555807479225739, 0.24385869673431038, 0.0985289555477556]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.1090352310148159, -0.24353771926105394, -0.16012495736153784, -0.08137129437606785]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.015885905750200907, 0.1561062169140983, 0.04185288495517297, 0.008583347231961257]

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

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.03896231840731087 - 0.16861580503614634im, -0.025289636874468526 - 0.17436278908639272im, 0.025289636874468526 - 0.17436278908639272im, 0.03896231840731087 - 0.16861580503614634im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.1275792672873622 + 4.6481249837923256e-5im, 0.4292747450667485 + 2.07975286867597e-5im, 0.2132531084776976 - 5.010572131564157e-5im, 0.09023321758757742 - 2.3324619293864605e-5im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.13045326117441555 + 0.06496897165512831im, -0.24884855418696092 + 0.02073400684859033im, -0.17997489419122087 - 0.05336375846201388im, -0.10051106154726067 - 0.06598907171238555im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.012591167963246538 + 8.342427765715567e-5im, 0.10000762643289299 + 2.8009239204921644e-5im, 0.031383769549874256 - 7.129336869851352e-5im, 0.006939177378416275 - 8.371772433941993e-5im]

end
