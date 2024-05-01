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
    μ = 0.0
    t1 = -0.25

    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta2.0_t-0.25_U2.089_mu1.0445_numc1_numk256"
    (; G0, G, Σ, Γ, occ) = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1), half_filling = true)
    plot_vertex_K1(Γ)
    plot_vertex_K2(Γ; vmax = 0.1)
    plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)

    G_new = copy(G)
    Dyson!(G_new, Σ, G0)
    @info absmax(G_new - G)
end;


begin
    T = 0.2
    U = 5.6
    μ = 2.1800201007694464 - U/2
    t1 = 1.0
    t2 = -0.3
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta5.0_t-1.0_U5.6_mu2.1800201007694464_numc1_numk255"
    (; G0, G, Σ, Γ, occ) = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1, t2), half_filling = false)

    close("all")
    plot_vertex_K1(Γ)
    plot_vertex_K2(Γ; vmax = 0.5)
    plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)

    G_new = copy(G)
    Dyson!(G_new, Σ, G0)
    @info absmax(G_new - G)
end

begin
    Γ = data_triqs.Γ
    Γ_mbe = fdDGAsolver.asymptotic_to_mbe(Γ);
    Ω = MatsubaraFrequency(T, 1, Boson)
    ν = MatsubaraFrequency(T, -1, Fermion)
    ω = MatsubaraFrequency(T, 1, Fermion)
    for Ch in [pCh, aCh, tCh], Sp in [pSp, xSp, dSp]
        @info Ch, Sp, abs(Γ(Ω, ν, ω, Ch, Sp) - Γ_mbe(Ω, ν, ω, Ch, Sp))
    end
    # @info Γ.F0(Ω, ν, ω, aCh, pSp) - U
end;



begin
    # Check parquet self-consistency
    nG  = div(length(G.meshes[1]), 2)
    nK1 = div(length(Γ.γa.K1.meshes[1]), 2) + 1
    nK2 = (div(length(Γ.γa.K2.meshes[1]), 2) + 1, div(length(Γ.γa.K2.meshes[2]), 2))
    nK3 = (1, 1)

    VT = Vertex
    # VT = MBEVertex

    # Set solver vertex and Green function to the one from TRIQS
    S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e=0., T, D=0., Δ=0., U, VT)
    if VT <: MBEVertex
        S.F = fdDGAsolver.asymptotic_to_mbe(Γ)
    else
        S.F = Γ
    end
    S.G = G
    bubbles!(S)

    # fdDGAsolver.BSE_K1!(S, pCh)
    # fdDGAsolver.BSE_K1!(S, aCh)
    # fdDGAsolver.BSE_K1!(S, tCh)
    # fdDGAsolver.BSE_K2!(S, pCh)
    # fdDGAsolver.BSE_K2!(S, aCh)
    # fdDGAsolver.BSE_K2!(S, tCh)

    fdDGAsolver.BSE_K1_new!(S, pCh)
    fdDGAsolver.BSE_K1_new!(S, aCh)
    fdDGAsolver.BSE_K1_new!(S, tCh)
    fdDGAsolver.BSE_K2_new!(S, pCh)
    fdDGAsolver.BSE_K2_new!(S, aCh)
    fdDGAsolver.BSE_K2_new!(S, tCh)

    @info absmax(S.Fbuff.γp.K1 - Γ.γp.K1)
    @info absmax(S.Fbuff.γa.K1 - Γ.γa.K1)
    @info absmax(S.Fbuff.γt.K1 - Γ.γt.K1)

    @info absmax(S.Fbuff.γp.K2 - Γ.γp.K2)
    @info absmax(S.Fbuff.γa.K2 - Γ.γa.K2)
    @info absmax(S.Fbuff.γt.K2 - Γ.γt.K2)


    for (K1, label) in zip([Γ.γp.K1, Γ.γt.K1, Γ.γa.K1], ["K1p", "K1t", "K1a"])
        plot(values(K1.meshes[1]), real.(K1.data), "o-"; label, alpha=0.5)
    end
    for (i, (K1, label)) in enumerate(zip([S.Fbuff.γp.K1, S.Fbuff.γt.K1, S.Fbuff.γa.K1], ["K1p", "K1t", "K1a"]))
        plot(values(K1.meshes[1]), real.(K1.data), "x-"; label, c="C$(i-1)")
    end
    xlim([-10, 10] .* (2π * temperature(Γ)))
    # ylim([-0.5, 0.5])
    legend()
    fig = gcf(); display(fig); close(fig)

    plot_vertex_K2(S.Fbuff; vmax = 0.01)
    mult_add!(S.Fbuff, Γ, -1)
    plot_vertex_K2(S.Fbuff; vmax = 0.0005)
end


begin
    (; G0, G, Σ, Γ, occ) = load_vertex_from_triqs(prefix, T, U; half_filling = false)

    nG  = div(length(G.meshes[1]), 2)
    nK1 = div(length(Γ.γa.K1.meshes[1]), 2) + 1
    nK2 = (div(length(Γ.γa.K2.meshes[1]), 2) + 1, div(length(Γ.γa.K2.meshes[2]), 2))
    nK3 = (1, 1)

    # Check parquet self-consistency
    S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e=0., T, D=0., Δ=0., U)

    # Set solver vertex and Green function to the one from TRIQS
    S.F = Γ
    S.G = G
    bubbles!(S)

    BSE_K1!(S, pCh)
    BSE_K1!(S, aCh)
    BSE_K1!(S, tCh)
    BSE_K2!(S, pCh)
    BSE_K2!(S, aCh)
    BSE_K2!(S, tCh)

    @info absmax(S.Fbuff.γp.K1 - Γ.γp.K1)
    @info absmax(S.Fbuff.γa.K1 - Γ.γa.K1)
    @info absmax(S.Fbuff.γt.K1 - Γ.γt.K1)

    @info absmax(S.Fbuff.γp.K2 - Γ.γp.K2)
    @info absmax(S.Fbuff.γa.K2 - Γ.γa.K2)
    @info absmax(S.Fbuff.γt.K2 - Γ.γt.K2)


    for (K1, label) in zip([Γ.γp.K1, Γ.γt.K1, Γ.γa.K1], ["K1p", "K1t", "K1a"])
        plot(values(K1.meshes[1]), real.(K1.data), "o-"; label, alpha=0.5)
    end
    for (i, (K1, label)) in enumerate(zip([S.Fbuff.γp.K1, S.Fbuff.γt.K1, S.Fbuff.γa.K1], ["K1p", "K1t", "K1a"]))
        plot(values(K1.meshes[1]), real.(K1.data), "x-"; label, c="C$(i-1)")
    end
    xlim([-10, 10] .* (2π * temperature(Γ)))
    # ylim([-0.5, 0.5])
    legend()
    fig = gcf(); display(fig); close(fig)

    plot_vertex_K2(Γ; vmax = 1.0)
    plot_vertex_K2(S.Fbuff; vmax = 1.0)
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
