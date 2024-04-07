using fdDGAsolver
using MatsubaraFunctions
using HDF5
using StaticArrays
using Test

@testset "NL2_Channel" begin
    using fdDGAsolver: kSW

    T = 0.5
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    γ = fdDGAsolver.NL2_Channel(T, 5, (4, 3), (2, 3), mK);
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
    k_ = BrillouinPoint(0, 5)
    q_ = BrillouinPoint(4, -2)
    P = fold_back(P_, mK)
    k = fold_back(k_, mK)
    q = fold_back(q_, mK)
    @test γ(Ω, ν, ω, P_, k_, q_)       ≈ γ.K1[Ω, P] + γ.K2[Ω, ν, P, k] + γ.K2[Ω, ω, P, q] + γ.K3[Ω, ν, ω, P]
    @test γ(Ω, νInf, ω, P_, k_, q_)    ≈ γ.K1[Ω, P] + γ.K2[Ω, ω, P, q]
    @test γ(Ω, ν, νInf, P_, k_, q_)    ≈ γ.K1[Ω, P] + γ.K2[Ω, ν, P, k]
    @test γ(Ω, νInf, νInf, P_, k_, q_) ≈ γ.K1[Ω, P]

    # Test reduce
    # For the K3 vertex, only the s-wave term is considered.
    x1 = γ(Ω, ν, ω, P, kSW, kSW; K1 = false, K2 = false, K3 = true)
    x2 = γ(Ω, νInf, ω, P, k, q; K1 = false, K2 = true, K3 = false)
    x3 = γ(Ω, ν, νInf, P, k, q; K1 = false, K2 = true, K3 = false)
    x4 = γ(Ω, νInf, νInf, P, k, q; K1 = true, K2 = false, K3 = false)
    fdDGAsolver.reduce!(γ)
    @test γ(Ω, ν, ω, P, kSW, kSW) ≈ x1
    @test γ(Ω, νInf, ω, P, k, q) ≈ x2
    @test γ(Ω, ν, νInf, P, k, q) ≈ x3
    @test γ(Ω, νInf, νInf, P, k, q) ≈ x4

    # Test IO
    testfile = dirname(@__FILE__) * "/test.h5"
    file = h5open(testfile, "w")
    save!(file, "f", γ)
    close(file)

    file = h5open(testfile, "r")
    γp = fdDGAsolver.load_channel(fdDGAsolver.NL2_Channel, file, "f")
    @test γ == γp
    @test γp isa fdDGAsolver.NL2_Channel
    close(file)

    rm(testfile; force=true)
end


@testset "NL2_Vertex" begin
    T = 0.5
    U = 3.0
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    F = fdDGAsolver.NL2_Vertex(RefVertex(T, U), T, 10, (4, 3), (2, 1), mK)
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
    Fp = fdDGAsolver.load_vertex(fdDGAsolver.NL2_Vertex, file, "f")
    @test F == Fp
    @test Fp isa fdDGAsolver.NL2_Vertex
    close(file)

    rm(testfile; force=true)
end

@testset "NL2 SWaveBrillouinPoint" begin
    # Test vertex evaluation using s-wave form factor kSW
    using fdDGAsolver: kSW

    T = 0.5
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    # Swave evaluation for NL2_Channel

    γ = fdDGAsolver.NL2_Channel(T, 5, (4, 3), (2, 3), mK);
    unflatten!(γ, rand(ComplexF64, length(γ)))

    Ω = MatsubaraFrequency(T, 1, Boson)
    ν = MatsubaraFrequency(T, 2, Fermion)
    ω = MatsubaraFrequency(T, -1, Fermion)
    k = BrillouinPoint(0, 1)
    q = BrillouinPoint(1, 1)

    function _test_swave(Γ, ωs...)
        # Compare Γ(ωs..., kSW) with a manual average over the momentum index
        mesh_P = meshes(Γ, length(ωs) + 1)
        val = mapreduce(+, mesh_P) do P
            Γ(ωs..., value(P))
        end / length(mesh_P)

        @test Γ[ωs..., kSW] ≈ val
    end
    _test_swave(γ.K1, Ω)
    _test_swave(γ.K3, Ω, ν, ω)

    # Test K2 separately because it has two momentum indices
    val = mapreduce(+, mK) do P
        γ.K2(Ω, ν, value(P), q)
    end / length(mK)
    @test γ.K2[Ω, ν, kSW, q] ≈ val

    val = mapreduce(+, mK) do P
        γ.K2(Ω, ν, k, value(P))
    end / length(mK)
    @test γ.K2[Ω, ν, k, kSW] ≈ val

    val = mapreduce(+, Iterators.product(mK, mK)) do (k, q)
        γ.K2(Ω, ν, value(k), value(q))
    end / length(mK)^2
    @test γ.K2[Ω, ν, kSW, kSW] ≈ val

    for ν in [MatsubaraFrequency(T, 2, Fermion), νInf]
        for ω in [MatsubaraFrequency(T, -1, Fermion), νInf]

            val1 = mapreduce(+, mK) do P
                γ(Ω, ν, ω, value(P), k, q)
            end / length(mK)

            val2 = mapreduce(+, mK) do P
                γ(Ω, ν, ω, k, value(P), q)
            end / length(mK)

            val3 = mapreduce(+, mK) do P
                γ(Ω, ν, ω, k, q, value(P))
            end / length(mK)
            val4 = mapreduce(+, Iterators.product(mK, mK, mK)) do (P, k, q)
                γ(Ω, ν, ω, value(P), value(k), value(q))
            end / length(mK)^3

            @test γ(Ω, ν, ω, kSW, k,   q  ) ≈ val1
            @test γ(Ω, ν, ω, k,   kSW, q  ) ≈ val2
            @test γ(Ω, ν, ω, k,   q,   kSW) ≈ val3
            @test γ(Ω, ν, ω, kSW, kSW, kSW) ≈ val4
        end
    end

    # Swave evaluation for NL2_Vertex

    F = fdDGAsolver.NL2_Vertex(RefVertex(T, 2.), T, 10, (4, 3), (2, 1), mK)
    unflatten!(F, rand(ComplexF64, length(F)))

    for P_ in mK, k_ in mK
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
                val3 = mapreduce(+, Iterators.product(mK, mK)) do (k, q)
                    F(Ω, ν, ω, P, value(k), value(q), Ch, Sp; γa, γp, γt, F0)
                end / length(mK)^2

                @test F(Ω, ν, ω, P, kSW, k,   Ch, Sp; γa, γp, γt, F0) ≈ val1
                @test F(Ω, ν, ω, P, k,   kSW, Ch, Sp; γa, γp, γt, F0) ≈ val2
                @test F(Ω, ν, ω, P, kSW, kSW, Ch, Sp; γa, γp, γt, F0) ≈ val3
            end
        end
    end
end
