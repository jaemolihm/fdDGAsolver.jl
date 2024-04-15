using fdDGAsolver
using MatsubaraFunctions
using HDF5
using StaticArrays
using Test

@testset "nonlocal bubbles" begin

    # Test bubbles are correct when Π and G have different BZ mesh

    T = 0.5
    μ = 0.2
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

    fdDGAsolver.bubbles_momentum_space!(Πpp1, Πph1, G1)
    fdDGAsolver.bubbles_momentum_space!(Πpp2, Πph2, G2)

    @test Πpp1 isa fdDGAsolver.NL2_MF_Π
    @test Πpp1.data ≈ Πpp2.data
    @test Πph1.data ≈ Πph2.data

    # Bubble without fermionic momentum dependence
    # Check this agrees with manual integration of the fermionic momentum
    Πpp0 = MeshFunction(mΠΩ, mΠν, mK_Π)
    Πph0 = copy(Πpp0)

    fdDGAsolver.bubbles_real_space!(Πpp0, Πph0, G1)
    fdDGAsolver.bubbles_real_space!(Πpp1, Πph1, G1)

    @test Πpp0 isa fdDGAsolver.NL_MF_Π
    @test Πpp0.data ≈ dropdims(sum(Πpp1.data, dims=4), dims=4) ./ length(mK_Π)
    @test Πph0.data ≈ dropdims(sum(Πph1.data, dims=4), dims=4) ./ length(mK_Π)

end


@testset "NL_Channel" begin
    using fdDGAsolver: kSW

    T = 0.5
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    γ = fdDGAsolver.NL_Channel(T, 5, (4, 3), (2, 3), mK);
    @test fdDGAsolver.numK1(γ) == 5
    @test fdDGAsolver.numK2(γ) == (4, 3)
    @test fdDGAsolver.numK3(γ) == (2, 3)
    @test fdDGAsolver.numP(γ) == length(mK)

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


    # Test evaluation with 3 momentum arguments
    # NL_Channel has no fermionic frequency dependence.

    k1 = BrillouinPoint(0, 0)
    k2 = BrillouinPoint(0, 0)
    for ν in [MatsubaraFrequency(T, 2, Fermion), νInf]
        for ω in [MatsubaraFrequency(T, -1, Fermion), νInf]
            for (K1, K2, K3) in Iterators.product((true, false), (true, false), (true, false))
                @test γ(Ω, ν, ω, P_, k1, k2; K1, K2, K3) == γ(Ω, ν, ω, P_; K1, K2, K3)
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
    γp = fdDGAsolver.load_channel(fdDGAsolver.NL_Channel, file, "f")
    @test γ == γp
    @test γp isa fdDGAsolver.NL_Channel
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

    # Test IO
    testfile = dirname(@__FILE__) * "/test.h5"
    file = h5open(testfile, "w")
    save!(file, "f", F)
    close(file)

    file = h5open(testfile, "r")
    Fp = fdDGAsolver.load_vertex(fdDGAsolver.NL_Vertex, file, "f")
    @test F == Fp
    @test Fp isa fdDGAsolver.NL_Vertex
    close(file)

    rm(testfile; force=true)
end

@testset "NL SWaveBrillouinPoint" begin
    # Test vertex evaluation using s-wave form factor kSW
    using fdDGAsolver: kSW

    T = 0.5
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    # Swave evaluation for NL_Channel

    γ = fdDGAsolver.NL_Channel(T, 5, (4, 3), (2, 3), mK);
    γ.K1.data .= rand(ComplexF64, size(γ.K1.data)...)
    γ.K2.data .= rand(ComplexF64, size(γ.K2.data)...)
    γ.K3.data .= rand(ComplexF64, size(γ.K3.data)...)
    Ω = MatsubaraFrequency(T, 1, Boson)
    ν = MatsubaraFrequency(T, 2, Fermion)
    ω = MatsubaraFrequency(T, -1, Fermion)

    function _test_swave(Γ, ωs...)
        # Compare Γ(ωs..., kSW) with a manual average over the momentum index
        mesh_P = meshes(Γ, length(ωs) + 1)
        val = mapreduce(+, mesh_P) do P
            Γ(ωs..., value(P))
        end / length(mesh_P)

        @test Γ[ωs..., kSW] ≈ val
    end
    _test_swave(γ.K1, Ω)
    _test_swave(γ.K2, Ω, ν)
    _test_swave(γ.K3, Ω, ν, ω)

    for ν in [MatsubaraFrequency(T, 2, Fermion), νInf]
        for ω in [MatsubaraFrequency(T, -1, Fermion), νInf]
            k0 = BrillouinPoint(0, 0)

            val = mapreduce(+, mK) do P
                γ(Ω, ν, ω, value(P), k0, k0)
            end / length(mK)

            @test γ(Ω, ν, ω, kSW, k0, k0) ≈ val
        end
    end


    # Swave evaluation for NL_Vertex

    F = fdDGAsolver.NL_Vertex(RefVertex(T, 2.), T, 10, (4, 3), (2, 1), mK)
    unflatten!(F, rand(ComplexF64, length(flatten(F))))

    ν0 = MatsubaraFrequency(T, 2, Fermion)
    ω0 = MatsubaraFrequency(T, -1, Fermion)
    for P_ in [mK[1], mK[8]], k_ in [mK[1], mK[4]], ν in [ν0, νInf], ω in [ω0, νInf]
        P = value(P_)
        k = value(k_)

        for Ch in (aCh, pCh, tCh), Sp in (pSp, xSp, dSp)
            for (γa, γp, γt, F0) in [
                (true, true, true, true),
                (true, false, false, false),
                (false, true, false, false),
                (false, false, true, false),
                (false, false, false, true),
            ]
                val1 = mapreduce(+, mK) do q
                    F(Ω, ν, ω, P, value(q), k, Ch, Sp; γa, γp, γt, F0)
                end / length(mK)
                val2 = mapreduce(+, mK) do q
                    F(Ω, ν, ω, P, k, value(q), Ch, Sp; γa, γp, γt, F0)
                end / length(mK)

                @test F(Ω, ν, ω, P, kSW, k, Ch, Sp; γa, γp, γt, F0) ≈ val1
                @test F(Ω, ν, ω, P, k, kSW, Ch, Sp; γa, γp, γt, F0) ≈ val2
            end
        end
    end
end
