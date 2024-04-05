using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using Test

@testset "Hubbard model" begin
    using MPI
    MPI.Init()

    T = 0.5
    t1 = 1.0
    μ = 0.2

    nG = 10
    mG = MatsubaraMesh(T, nG, Fermion)

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(8, k1, k2))

    # Test bare Green function of the Hubbard model

    Gbare = hubbard_bare_Green(mG, mK; μ, t1)

    for ν in 2π * T * ((-2:2) .+ 1/2)
        @test Gbare(ν, SVector(0., 0.)  ) ≈ 1 / (im * ν + μ + 4t1)
        @test Gbare(ν, SVector(0., π/2) ) ≈ 1 / (im * ν + μ + 2t1)
        @test Gbare(ν, SVector(0., π)   ) ≈ 1 / (im * ν + μ + 0t1)
        @test Gbare(ν, SVector(0., 3π/2)) ≈ 1 / (im * ν + μ + 2t1)
        @test Gbare(ν, SVector{2, Float64}(π, π)) ≈ 1 / (im * ν + μ - 4t1)
        @test Gbare(ν, SVector(0.2 + 2π, 0.4 - 2π)) ≈ Gbare(ν, SVector(0.2, 0.4))
    end

end
