using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using LinearAlgebra
using Test
using PyPlot
using HDF5
using MatsubaraFunctions: mesh_index
using fdDGAsolver: numP_Γ, k0, kSW

include("/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/script/plot_vertex.jl")

begin
    T = 0.5
    U = 2.089
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta2.0_t-0.25_U2.0809_mu1.04045_numc1_numk256"
    Γ = load_vertex_from_triqs(prefix, T, U; half_filling = true)
    plot_vertex_K1(Γ)
    plot_vertex_K2(Γ; vmax = 0.01)
    plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)
end
begin
    T = 0.2
    U = 5.6
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta5.0_t-1.0_U5.6_mu2.1800201007694464_numc1_numk255"
    Γ = load_vertex_from_triqs(prefix, T, U; half_filling = false)

    close("all")
    plot_vertex_K1(Γ)
    plot_vertex_K2(Γ; vmax = 0.5)
    plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)
end



begin
    # Check the local Green fucntion G can be reproduced by the lattice Dyson equation
    T = 0.5
    μ = 0.
    t1 = -0.25
    t2 = 0.0
    U = 2.089

    T = 0.2
    t1 = 1.0
    t2 = -0.3
    U = 5.6
    μ = 2.18 - U/2
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    nG  = 128
    mK_G = BrillouinZoneMesh(BrillouinZone(64, k1, k2))

    mG = MatsubaraMesh(T, nG, Fermion)
    Gbare = hubbard_bare_Green(mG, mK_G; μ, t1, t2)
    Σ0 = copy(Gbare)
    set!(Σ0, 0)
    for ν in MatsubaraFunctions.meshes(Σ0, Val(1))
        view(Σ0, ν, :) .= Σ[value(ν)]
    end
    G_from_Dyson = copy(Gbare)
    Dyson!(G_from_Dyson, Σ0, Gbare)

    νs = MatsubaraFrequency.(T, -33:32, Fermion)
    @info "G_TRIQS - G_Dyson = ", norm(G.(νs) .- G_from_Dyson.(νs, Ref(kSW)))

    νs = MatsubaraFrequency.(T, -33:32, Fermion)
    plot(value.(νs), real.(G.(νs)))
    plot(value.(νs), real.(G_from_Dyson.(νs, Ref(kSW))), "--")

    plot(value.(νs), imag.(G.(νs)))
    plot(value.(νs), imag.(G_from_Dyson.(νs, Ref(kSW))), "--")
    xlim([-10, 10])
    fig = gcf(); display(fig); close(fig)
end
