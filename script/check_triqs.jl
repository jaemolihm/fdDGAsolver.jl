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
    filename = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point_U6.2.h5"
    filename = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point_U6.0.h5"
    filename = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point_U5.8.h5"
    filename = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point_U5.6.h5"
    filename = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point_U5.4.h5"
    filename = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point_U5.2.h5"
    filename = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point_U5.0.h5"
    filename = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point.h5"
end

begin
    # Check parquet self-consistency of the local DMFT vertex
    data_triqs = load_triqs_data(filename);
    (; G, Σ, Γ) = data_triqs
    (; U, T, μ, t1, t2, t3) = data_triqs.params

    display(Γ)

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

    mult_add!(S.Fbuff, Γ, -1)
    # plot_vertex_K2(Γ; vmax = 1.0)
    plot_vertex_K2(S.Fbuff; vmax = 0.1)
end


begin
    # Check DMFT self-consistency of the local Green function
    data_triqs = load_triqs_data(filename)
    (; G0, G, Σ, occ) = data_triqs
    (; U, T, μ, t1, t2, t3) = data_triqs.params

    # Check Dyson equation within the impurity
    G_new = copy(G)
    Dyson!(G_new, Σ, G0)
    @info "Impurity Dyson error : $(absmax(G_new - G))"

    # Check occupation
    @info "Occupation from file = $(data_triqs.occ)"
    @info "Occupation from G    = $(compute_occupation(G))"

    nG  = 128
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
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
