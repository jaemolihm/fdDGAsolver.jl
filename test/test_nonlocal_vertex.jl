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
end
