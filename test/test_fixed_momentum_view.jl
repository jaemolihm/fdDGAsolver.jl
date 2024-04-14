using fdDGAsolver
using MatsubaraFunctions
using HDF5
using StaticArrays
using Test

@testset "fixed_momentum_view" begin
    T = 0.5
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))

    F1 = fdDGAsolver.Vertex(RefVertex(T, 2.), T, 10, (4, 4), (3, 2))
    unflatten!(F1, rand(ComplexF64, length(flatten(F1))))

    F2 = fdDGAsolver.NL_Vertex(F1, T, 8, (4, 4), (3, 2), mK)
    unflatten!(F2, rand(ComplexF64, length(flatten(F2))))

    F3 = fdDGAsolver.NL2_Vertex(F2, T, 5, (3, 3), (2, 2), mK)
    unflatten!(F3, rand(ComplexF64, length(flatten(F3))))

    Ω = MatsubaraFrequency(T, -1, Boson)
    ν0 = MatsubaraFrequency(T, -1, Fermion)
    ω0 = MatsubaraFrequency(T, 1, Fermion)
    P = BrillouinPoint(-1, 2)
    k = BrillouinPoint(2, 7)
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
                @test F1view(Ω, ν, ω, Ch, Sp; F0, γt, γp, γa) ≈ F1(Ω, ν, ω, P, k, q, Ch, Sp; F0, γt, γp, γa)
                @test F2view(Ω, ν, ω, Ch, Sp; F0, γt, γp, γa) ≈ F2(Ω, ν, ω, P, k, q, Ch, Sp; F0, γt, γp, γa)
                @test F3view(Ω, ν, ω, Ch, Sp; F0, γt, γp, γa) ≈ F3(Ω, ν, ω, P, k, q, Ch, Sp; F0, γt, γp, γa)
            end
        end
    end
end
