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

        res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = mode);

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.05172399220018011, -0.0382726983425996, 0.0382726983425996, 0.05172399220018011]
        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.14100149828795813, 0.5564902790134902, 0.2441845526441227, 0.0986585584531646]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10914654884487347, -0.24367152908284298, -0.16026257759686086, -0.0814591544081497]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.015924271701333197, 0.15638058247174064, 0.04194694271003733, 0.008604246531834949]

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

    res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = :threads);

    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.03823850779995448 - 0.1680218349391556im, -0.025007279341553813 - 0.17376193156566136im, 0.025007279341553813 - 0.17376193156566136im, 0.03823850779995448 - 0.1680218349391556im]
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.12769365574194508 + 4.669583083272952e-5im, 0.4293472788847892 + 2.0889557241467297e-5im, 0.21341117339581095 - 5.032906796097789e-5im, 0.09031671909124052 - 2.3434599421452944e-5im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.13065162376702708 + 0.06508944506459405im, -0.2489998847943091 + 0.020761158463655182im, -0.18018087600352709 - 0.053440120236298495im, -0.10068306532398233 - 0.06612909654020642im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.012608817961964197 + 8.379814938074763e-5im, 0.10000656344261566 + 2.81341808792915e-5im, 0.03141571004131562 - 7.161170753431479e-5im, 0.006949633916214235 - 8.408875754509944e-5im]

end
