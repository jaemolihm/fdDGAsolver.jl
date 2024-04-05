using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Test

@testset "NL_Channel" begin
    T = 0.5
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    γ = fdDGAsolver.NL_Channel(T, 5, (4, 3), (2, 3), mK);
    γ.K1.data .= rand(ComplexF64, size(γ.K1.data)...)
    γ.K2.data .= rand(ComplexF64, size(γ.K2.data)...)
    γ.K3.data .= rand(ComplexF64, size(γ.K3.data)...)
    Ω = MatsubaraFrequency(T, 1, Boson)
    ν = MatsubaraFrequency(T, 2, Fermion)
    ω = MatsubaraFrequency(T, -1, Fermion)
    P_ = BrillouinPoint(-1, 1)
    P = fold_back(P_, mK)
    @test γ(Ω, ν, ω, P_)       ≈ γ.K1[Ω, P] + γ.K2[Ω, ν, P] + γ.K2[Ω, ω, P] + γ.K3[Ω, ν, ω, P]
    @test γ(Ω, νInf, ω, P_)    ≈ γ.K1[Ω, P] + γ.K2[Ω, ω, P]
    @test γ(Ω, ν, νInf, P_)    ≈ γ.K1[Ω, P] + γ.K2[Ω, ν, P]
    @test γ(Ω, νInf, νInf, P_) ≈ γ.K1[Ω, P]

    # Test reduce
    x1 = γ(Ω, ν, ω, P; K1 = false, K2 = false, K3 = true)
    x2 = γ(Ω, νInf, ω, P; K1 = false, K2 = true, K3 = false)
    x3 = γ(Ω, ν, νInf, P; K1 = false, K2 = true, K3 = false)
    x4 = γ(Ω, νInf, νInf, P; K1 = true, K2 = false, K3 = false)
    fdDGAsolver.reduce!(γ)
    @test γ(Ω, ν, ω, P) ≈ x1
    @test γ(Ω, νInf, ω, P) ≈ x2
    @test γ(Ω, ν, νInf, P) ≈ x3
    @test γ(Ω, νInf, νInf, P) ≈ x4

    # Test IO
    testfile = dirname(@__FILE__) * "/test.h5"
    file = h5open(testfile, "w")
    save!(file, "f", γ)
    close(file)

    file = h5open(testfile, "r")
    γp = fdDGAsolver.load_nonlocal_channel(file, "f")
    @test γ == γp
    close(file)

    rm(testfile; force=true)
end
