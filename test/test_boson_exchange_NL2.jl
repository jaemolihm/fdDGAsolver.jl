using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using HDF5
using Test

@testset "NL2_MBEVertex" begin
    using MPI
    MPI.Init()
    using fdDGAsolver: NL2_MBEVertex, k0

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
end;
