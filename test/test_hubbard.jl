using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using Test

@testset "Hubbard model" begin
    using MPI
    MPI.Init()

    T = 0.5
    t1 = 1.3
    μ = 0.2

    nG = 5
    mG = MatsubaraMesh(T, nG, Fermion)

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(8, k1, k2))

    # Test bare Green function of the Hubbard model

    Gbare = hubbard_bare_Green(mG, mK; μ, t1)

    # Gbare stores im * G_physical, so we test -im * Gbare = G_physical
    for ν in 2π * T * ((-2:2) .+ 1/2)
        @test -im * Gbare(ν, SVector(0., 0.)  ) ≈ 1 / (im * ν + μ + 4t1)
        @test -im * Gbare(ν, SVector(0., π/2) ) ≈ 1 / (im * ν + μ + 2t1)
        @test -im * Gbare(ν, SVector(0., π)   ) ≈ 1 / (im * ν + μ + 0t1)
        @test -im * Gbare(ν, SVector(0., 3π/2)) ≈ 1 / (im * ν + μ + 2t1)
        @test -im * Gbare(ν, SVector{2, Float64}(π, π)) ≈ 1 / (im * ν + μ - 4t1)
        @test Gbare(ν, SVector(0.2 + 2π, 0.4 - 2π)) ≈ Gbare(ν, SVector(0.2, 0.4))
    end

    # Test creation of bubbles

    mΠΩ = MatsubaraMesh(T, 4, Boson)
    mΠν = MatsubaraMesh(T, 8, Fermion)

    Πpp = MeshFunction(mΠΩ, mΠν, mK, mK)
    Πph = MeshFunction(mΠΩ, mΠν, mK, mK)

    fdDGAsolver.bubbles!(Πpp, Πph, Gbare)

    # Test Dyson

    G = MeshFunction(mG, mK)
    Σ = MeshFunction(mG, mK)

    set!(Σ, 0)
    fdDGAsolver.Dyson!(G, Σ, Gbare)
    @test absmax(G - Gbare) < 1e-10

    set!(Σ, -0.5 + 0.2im)
    fdDGAsolver.Dyson!(G, Σ, Gbare)
    ν, k = π * T, SVector(π/2, 0.)
    @test G(ν, k) ≈ 1 / (1 / Gbare(ν, k) + Σ(ν, k))

    # We store im * G and im * Σ in the variables G and Σ. Hence, the variables G and Σ
    # should satisfy the ordinary Dyson equation G⁻¹ = Gbare⁻¹ - Σ when multiplied by -im.
    @test 1 / (-im * G(ν, k)) ≈ 1 / (-im * Gbare(ν, k)) - (-im * Σ(ν, k))


    # Test bubbles

    Gbare = hubbard_bare_Green(mG, mK; μ, t1)

    Πpp = MeshFunction(mΠΩ, mΠν, mK, mK)
    Πph = MeshFunction(mΠΩ, mΠν, mK, mK)
    fdDGAsolver.bubbles!(Πpp, Πph, Gbare)

    Ω = MatsubaraFrequency(T, 3, Boson)
    ν = MatsubaraFrequency(T, -2, Fermion)
    P = value(mK[58])
    k = value(mK[23])
    @test Πpp(Ω, ν, P, k) ≈ Gbare(ν, k) * Gbare(Ω - ν, P - k)
    @test Πph(Ω, ν, P, k) ≈ Gbare(Ω + ν, P + k) * Gbare(ν, k)

end
