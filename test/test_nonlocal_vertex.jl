using fdDGAsolver
using MatsubaraFunctions
using HDF5
using StaticArrays
using Test

@testset "NL bubbles" begin

    # Test bubbles are correct when Π and G have different BZ mesh

    T = 0.5
    μ = 0.
    t1 = 1.0

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    mG  = MatsubaraMesh(T, 5, Fermion)
    mΠΩ = MatsubaraMesh(T, 5, Boson)
    mΠν = MatsubaraMesh(T, 5, Fermion)

    mK_G = BrillouinZoneMesh(BrillouinZone(6, k1, k2))
    mK_Π = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    G1 = hubbard_bare_Green(mG, mK_G; μ, t1)
    G2 = hubbard_bare_Green(mG, mK_Π; μ, t1)

    Πpp1 = MeshFunction(mΠΩ, mΠν, mK_Π, mK_Π)
    Πph1 = copy(Πpp1)
    Πpp2 = copy(Πpp1)
    Πph2 = copy(Πpp1)

    fdDGAsolver.bubbles!(Πpp1, Πph1, G1)
    fdDGAsolver.bubbles!(Πpp2, Πph2, G2)

    @test Πpp1.data ≈ Πpp2.data
    @test Πph1.data ≈ Πph2.data

end


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

    @test fdDGAsolver.numK1(γ) == 5
    @test fdDGAsolver.numK2(γ) == (4, 3)
    @test fdDGAsolver.numK3(γ) == (2, 3)
    @test fdDGAsolver.numP(γ) == length(mK)

    # Test evaluation with 3 momentum arguments
    # NL_Channel has no fermionic frequency dependence.

    k1 = BrillouinPoint(0, 0)
    k2 = BrillouinPoint(0, 0)
    for ν in [MatsubaraFrequency(T, 2, Fermion), νInf]
        for ω in [MatsubaraFrequency(T, -1, Fermion), νInf]
            for (K1, K2, K3) in Iterators.product((true, false), (true, false), (true, false))
                @test γ(Ω, ν, ω, P_; K1, K2, K3) == γ(Ω, ν, ω, P_, k1, k2; K1, K2, K3)
            end
        end
    end

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


@testset "NL_Vertex" begin
    T = 0.5
    U = 3.0
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    F = fdDGAsolver.NL_Vertex(RefVertex(T, U), T, 10, (4, 3), (2, 1), mK)
    set!(F, 0)
    F.γp.K1.data .= rand(ComplexF64, size(F.γp.K1.data)...)
    F.γt.K1.data .= rand(ComplexF64, size(F.γt.K1.data)...)
    F.γa.K1.data .= rand(ComplexF64, size(F.γa.K1.data)...)

    @test fdDGAsolver.numK1(F) == 10
    @test fdDGAsolver.numK2(F) == (4, 3)
    @test fdDGAsolver.numK3(F) == (2, 1)
    @test fdDGAsolver.numP(F) == length(mK)

    Ω = MatsubaraFrequency(T, 1, Boson)
    ν = MatsubaraFrequency(T, 2, Fermion)
    ω = MatsubaraFrequency(T, -1, Fermion)
    P = BrillouinPoint(-1, 1)
    k = BrillouinPoint(1, 1)
    q = BrillouinPoint(0, 1)

    @test F(Ω, νInf, νInf, P, k, q, pCh, pSp) ≈ U + F.γp.K1(Ω, P)
    @test F(Ω, νInf, νInf, P, k, q, tCh, pSp) ≈ U + F.γt.K1(Ω, P)
    @test F(Ω, νInf, νInf, P, k, q, aCh, pSp) ≈ U + F.γa.K1(Ω, P)

    @test F(Ω, ν, ω, P, k, q, pCh, pSp) ≈ U + F.γp.K1(Ω, P) + F.γt.K1(Ω - ν - ω, P - k - q) + F.γa.K1(ν - ω, k - q)
    @test F(Ω, ν, ω, P, k, q, tCh, pSp) ≈ U + F.γt.K1(Ω, P) + F.γp.K1(Ω + ν + ω, P + k + q) + F.γa.K1(ω - ν, q - k)
    @test F(Ω, ν, ω, P, k, q, aCh, pSp) ≈ U + F.γa.K1(Ω, P) + F.γp.K1(Ω + ν + ω, P + k + q) + F.γt.K1(ν - ω, k - q)
end
