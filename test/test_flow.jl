using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using Test

@testset "flow" begin
    data_triqs = load_triqs_data(joinpath(dirname(pathof(fdDGAsolver)), "../data/Wu_point.h5"))

    (; T, μ, t1, t2, t3) = data_triqs.params

    nG = 5
    mG = MatsubaraMesh(T, nG, Fermion)

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    mK = BrillouinZoneMesh(BrillouinZone(36, k1, k2))

    G0_imp = data_triqs.G0
    Σ_imp = data_triqs.Σ
    G0_lat = hubbard_bare_Green(mG, mK; μ, t1, t2, t3)

    # Test Λ = ∞ gives G0_Λ = G0_imp
    G0_Λ = bare_Green_Ω_flow(1e6, G0_imp, Σ_imp, G0_lat)
    for k in [BrillouinPoint(0, 0), BrillouinPoint(0, 1)]
        @test maximum(abs.(G0_Λ[:, k] - [G0_imp[ν] for ν in value.(mG)])) < 1e-8
    end

    # Test Λ = 0 gives G0_Λ = G0_lat
    G0_Λ = bare_Green_Ω_flow(0.0, G0_imp, Σ_imp, G0_lat)
    @test absmax(G0_Λ - G0_lat) < 1e-8
end
