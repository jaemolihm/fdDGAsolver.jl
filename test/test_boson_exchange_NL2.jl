using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using HDF5
using Test

@testset "NL2_MBEVertex" begin
    using MPI
    MPI.Init()
    using fdDGAsolver: k0, get_reducible_vertex, kSW

    T = 0.5
    U = 2.0
    numK1 = 10
    numK2 = (5, 5)
    numK3 = (3, 3)

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    Ω = MatsubaraFrequency(T, 0, Boson)
    ν = MatsubaraFrequency(T, 0, Fermion)
    ω = MatsubaraFrequency(T, 1, Fermion)
    P = BrillouinPoint(0, 1)
    k = BrillouinPoint(2, 1)
    q = BrillouinPoint(-1, 1)

    Γ = NL2_MBEVertex(RefVertex(T, U), T, numK1, numK2, numK3, mK)

    # Test bare vertex
    unflatten!(Γ, zeros(eltype(Γ), length(Γ)))
    for Ch in [aCh, pCh, tCh], (Sp, U_Sp) in zip([pSp, dSp, xSp], [U, U, -U])
        @test Γ(Ω, ν, ω, P, k, q, Ch, Sp) ≈ U_Sp
    end

    # Test InfiniteMatsubaraFrequency evaluation
    unflatten!(Γ, rand(eltype(Γ), length(Γ)))
    for (Ch, γ) in zip([aCh, pCh, tCh], [Γ.γa, Γ.γp, Γ.γt])
        @test Γ(Ω,    ν, νInf, P, k, q, Ch, pSp) ≈ U + γ.K1(Ω, P) + γ.K2(Ω, ν, P, k)
        @test Γ(Ω, νInf,    ω, P, k, q, Ch, pSp) ≈ U + γ.K1(Ω, P) + γ.K2(Ω, ω, P, q)
        @test Γ(Ω, νInf, νInf, P, k, q, Ch, pSp) ≈ U + γ.K1(Ω, P)
    end

    # Test copy
    Γ_copy = copy(Γ)
    @test Γ == Γ_copy
    @test Γ.F0 !== Γ_copy.F0
    @test Γ.γp !== Γ_copy.γp

    # Test flatten and unflatten
    set!(Γ_copy, 0)
    @test all(flatten(Γ_copy) .== 0)
    @test Γ_copy != Γ

    unflatten!(Γ_copy, flatten(Γ))
    @test Γ_copy == Γ

    # Test IO
    testfile = dirname(@__FILE__) * "/test.h5"
    file = h5open(testfile, "w")
    save!(file, "f", Γ)
    close(file)

    file = h5open(testfile, "r")
    Γ_read = fdDGAsolver.load_vertex(NL2_MBEVertex, file, "f")
    @test Γ_read == Γ
    @test Γ_read isa NL2_MBEVertex
    close(file)

    rm(testfile; force=true)


    # Test conversion between asymptotic and MBE vertices
    F = NL2_Vertex(RefVertex(T, U), T, numK1, numK2, numK3, mK)
    unflatten!(F, rand(ComplexF64, length(F)))

    F_mbe = asymptotic_to_mbe(F)

    Ω = MatsubaraFrequency(T, 0, Boson)
    ν = MatsubaraFrequency(T, -2, Fermion)
    ω = MatsubaraFrequency(T, 1, Fermion)
    P = BrillouinPoint(0, 1)
    k = BrillouinPoint(2, 1)
    q = BrillouinPoint(-1, 1)

    for Ch in (aCh, pCh, tCh), Sp in (pSp, dSp, xSp)
        # Asymptotic and MBE vertices equal only for the s wave component
        γa = Ch === aCh
        γp = Ch === pCh
        γt = Ch === tCh
        @test F_mbe(Ω, ν, ω, P, kSW, kSW, Ch, Sp; γa, γp, γt) ≈ F(Ω, ν, ω, P, kSW, kSW, Ch, Sp; γa, γp, γt)
    end

    F_new = mbe_to_asymptotic(F_mbe)
    @test flatten(F) ≈ flatten(F_new)


    # Swave evaluation for NL2_MBEVertex
    F = fdDGAsolver.NL2_MBEVertex(RefVertex(T, 2.), T, 10, (4, 3), (2, 1), mK)
    unflatten!(F, rand(ComplexF64, length(F)))

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
                val3 = mapreduce(+, Iterators.product(mK, mK)) do (k, q)
                    F(Ω, ν, ω, P, value(k), value(q), Ch, Sp; γa, γp, γt, F0)
                end / length(mK)^2

                @test F(Ω, ν, ω, P, kSW, k,   Ch, Sp; γa, γp, γt, F0) ≈ val1
                @test F(Ω, ν, ω, P, k,   kSW, Ch, Sp; γa, γp, γt, F0) ≈ val2
                @test F(Ω, ν, ω, P, kSW, kSW, Ch, Sp; γa, γp, γt, F0) ≈ val3
            end
        end
    end
end;


@testset "NL2_MBEVertex fixed_momentum_view" begin
    T = 0.5
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    F1 = fdDGAsolver.MBEVertex(RefVertex(T, 2.), T, 10, (8, 8), (6, 6))
    unflatten!(F1, rand(ComplexF64, length(flatten(F1))))

    # F2 = fdDGAsolver.NL_Vertex(F1, T, 8, (4, 4), (3, 2), mK)
    # unflatten!(F2, rand(ComplexF64, length(flatten(F2))))
    F2 = F1

    F3 = fdDGAsolver.NL2_MBEVertex(F2, T, 6, (5, 5), (4, 4), mK)
    unflatten!(F3, rand(ComplexF64, length(flatten(F3))))

    Ω = MatsubaraFrequency(T, -1, Boson)
    ν0 = MatsubaraFrequency(T, -1, Fermion)
    ω0 = MatsubaraFrequency(T, 1, Fermion)
    P = BrillouinPoint(-1, 2)
    k = BrillouinPoint(2, 6)
    q = BrillouinPoint(1, -2)

    for Ch in (aCh, pCh, tCh)
        F1view = fdDGAsolver.fixed_momentum_view(F1, P, k, q, Ch)
        F2view = fdDGAsolver.fixed_momentum_view(F2, P, k, q, Ch)
        F3view = fdDGAsolver.fixed_momentum_view(F3, P, k, q, Ch)
        for Sp in (pSp, xSp, dSp), ν in [ν0, νInf], ω in [ω0, νInf]
            for (γa, γp, γt, F0) in [
                (true, true, true, true),
                (true, false, false, false),
                (false, true, false, false),
                (false, false, true, false),
                (false, false, false, true),
            ]
                # @info Ch, Sp, ν, ω, γa, γp, γt, F0
                @test F1view(Ω, ν, ω, Ch, Sp; F0, γt, γp, γa) ≈ F1(Ω, ν, ω, P, k, q, Ch, Sp; F0, γt, γp, γa)
                @test F2view(Ω, ν, ω, Ch, Sp; F0, γt, γp, γa) ≈ F2(Ω, ν, ω, P, k, q, Ch, Sp; F0, γt, γp, γa)
                @test F3view(Ω, ν, ω, Ch, Sp; F0, γt, γp, γa) ≈ F3(Ω, ν, ω, P, k, q, Ch, Sp; F0, γt, γp, γa)
            end
        end
    end
end
