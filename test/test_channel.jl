using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Test

@testset "Channel" begin
    T = 0.5
    numK1 = 10
    numK2 = (5, 5)
    numK3 = (3, 3)
    γ = fdDGAsolver.Channel(T, numK1, numK2, numK3)
    @test MatsubaraFunctions.temperature(γ) == T
    @test fdDGAsolver.numK1(γ) == numK1
    @test fdDGAsolver.numK2(γ) == numK2
    @test fdDGAsolver.numK3(γ) == numK3
    @test length(γ) == (2numK1 - 1) + (2numK2[1] - 1) * 2numK2[2] + (2numK3[1] - 1) * (2numK3[2])^2
    @test length(flatten(γ)) == length(γ)

    # Test copy
    γ.K1.data .= rand(size(γ.K1.data)...)
    γ.K2.data .= rand(size(γ.K2.data)...)
    γ.K3.data .= rand(size(γ.K3.data)...)
    γ_copy = copy(γ)
    @test γ == γ_copy

    set!(γ_copy.K1, 0)
    set!(γ_copy.K2, 0)
    set!(γ_copy.K3, 0)
    @test γ != γ_copy
    @test MatsubaraFunctions.absmax(γ) > 0
    @test MatsubaraFunctions.absmax(γ_copy) == 0

    # Test flatten and unflatten
    unflatten!(γ_copy, flatten(γ))
    @test γ == γ_copy

    # Test reduce
    x1 = γ(Ω, ν, ω; K1 = false, K2 = false, K3 = true)
    x2 = γ(Ω, νInf, ω; K1 = false, K2 = true, K3 = false)
    x3 = γ(Ω, ν, νInf; K1 = false, K2 = true, K3 = false)
    x4 = γ(Ω, νInf, νInf; K1 = true, K2 = false, K3 = false)
    fdDGAsolver.reduce!(γ)
    @test γ(Ω, ν, ω) ≈ x1
    @test γ(Ω, νInf, ω) ≈ x2
    @test γ(Ω, ν, νInf) ≈ x3
    @test γ(Ω, νInf, νInf) ≈ x4

    # Test evaluation
    for νInf in [MatsubaraFrequency(T, 10^10, Fermion), fdDGAsolver.νInf]
        Ω  = value(meshes(γ.K3, 1)[5])
        ν1 = value(meshes(γ.K3, 2)[3])
        ν2 = value(meshes(γ.K3, 3)[4])
        @test γ(Ω, ν1,   ν2)   ≈ γ.K1[Ω] + γ.K2[Ω, ν1] + γ.K2[Ω, ν2] + γ.K3[Ω, ν1, ν2]
        @test γ(Ω, νInf, νInf) ≈ γ.K1[Ω]
        @test γ(Ω, ν1,   νInf) ≈ γ.K1[Ω] + γ.K2[Ω, ν1]
        @test γ(Ω, νInf, ν2)   ≈ γ.K1[Ω] + γ.K2[Ω, ν2]
    end
    @test fdDGAsolver.νInf === fdDGAsolver.InfiniteMatsubaraFrequency()


    for iΩ in [-100, -1, 0, 1, 100], iω in [-100, -1, 0, 1, 100]
        Ω = MatsubaraFrequency(T, iΩ, Boson);
        ω = MatsubaraFrequency(T, iω, Fermion);
        @test fdDGAsolver.box_eval(γ.K2, Ω, ω) ≈ γ(Ω, ω, νInf; K1 = false)
    end

    # Test IO
    testfile = dirname(@__FILE__) * "/test.h5"
    file = h5open(testfile, "w")
    save!(file, "f", γ)
    close(file)

    file = h5open(testfile, "r")
    γp = fdDGAsolver.load_channel(file, "f")
    @test γ == γp
    close(file)

    rm(testfile; force=true)
end
