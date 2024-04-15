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

prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta2.0_t-0.25_U2.0809_mu1.04045_numc1_numk256"
filename_G = "$(prefix)_DCA.h5"
filename_chi = "$(prefix)_chi3_fromDCA.h5"
filename_F = "$(prefix)_F_fromDCA.h5"

function _drop_first_dim(f)
    MeshFunction(f.meshes[2:end], dropdims(f.data, dims=1))
end

begin
    T = 0.5
    U = 2.089
    Ω0 = MatsubaraFrequency(T, 0, Boson)

    #-----------------------------------------------------------------------------#
    # Green function and self-energy

    f = h5open(filename_G, "r")
    occ = h5read(filename_G, "density")
    G = _drop_first_dim(load_triqs_gf(f, "cG_k_iw"));
    Σ = _drop_first_dim(load_triqs_gf(f, "cSigma_k_iw"));
    close(f)

    # Subtract U/2 because we put Σ = 0 at half filling and in TRIQS Σ = U/2 is half filling.
    Σ.data .-= U/2

    # We factor out -im from G and Σ and store im * G and im * Σ
    mult!(G, im)
    mult!(Σ, im)

    #-----------------------------------------------------------------------------#
    # K1 vertex

    # Read K1 vertex
    f = h5open(filename_chi, "r")
    χD = _drop_first_dim(load_triqs_gf(f, "chi2_D_k"));
    χM = _drop_first_dim(load_triqs_gf(f, "chi2_M_k"));
    χS = _drop_first_dim(load_triqs_gf(f, "chi2_S_k"));
    close(f)

    # Subtract disconnected part
    χD[Ω0] -= occ^2 * 2 / T

    # from χ2 compute K1
    K1D = copy(χD)
    K1M = copy(χM)
    K1S = copy(χS)
    mult!(K1D, -U^2)
    mult!(K1M, -U^2)
    mult!(K1S, -U^2)

    #-----------------------------------------------------------------------------#
    # K2 vertex

    f = h5open(filename_chi, "r")
    χ3D = load_triqs_gf(f, "chi3_D_k") |> _drop_first_dim |> _drop_first_dim;
    χ3M = load_triqs_gf(f, "chi3_M_k") |> _drop_first_dim |> _drop_first_dim;
    χ3S = load_triqs_gf(f, "chi3_S_k") |> _drop_first_dim |> _drop_first_dim;
    close(f)

    # Subtract disconnected part (factor 2 comes from the spin structure)
    # (-i) because we factor out (-i) from G
    for ν in meshes(χ3D, Val(2))
        χ3D[Ω0, ν] -= 2 * occ * G(value(ν)) * -im / T
    end

    # from χ3 and K1 compute K2
    K2D = copy(χ3D)
    K2M = copy(χ3M)
    K2S = copy(χ3S)

    for ν in meshes(K2D, Val(2)), Ω in meshes(K2D, Val(1))
        # minus sign because we factor out (-i) from G
        K2D[Ω, ν] =  U * χ3D[Ω, ν] / G(value(ν)) / G(value(Ω + ν)) - K1D(value(Ω)) - U
        K2M[Ω, ν] = -U * χ3M[Ω, ν] / G(value(ν)) / G(value(Ω + ν)) - K1M(value(Ω)) + U
        K2S[Ω, ν] = -U * χ3S[Ω, ν] / G(value(ν)) / G(value(Ω - ν)) - K1S(value(Ω)) - 2U
    end
end;

begin
    #-----------------------------------------------------------------------------#
    # Core vertex

    f = h5open(filename_F, "r")
    F_D = load_triqs_gf(f, "F_D_k") |> _drop_first_dim |> _drop_first_dim |> _drop_first_dim
    F_M = load_triqs_gf(f, "F_M_k") |> _drop_first_dim |> _drop_first_dim |> _drop_first_dim
    F_S = load_triqs_gf(f, "F_S_k") |> _drop_first_dim |> _drop_first_dim |> _drop_first_dim
    close(f)

    mult!(F_D, -1)
    mult!(F_M, -1)
    mult!(F_S, -1)

    Ft_p = copy(F_D)
    set!(Ft_p, 0)
    Ft_x = copy(Ft_p)
    Fp_p = copy(Ft_p)
    Fp_x = copy(Ft_p)

    # ``S = pSp - xSp``
    # ``T = pSp + xSp``
    # ``D = 2 * pSp + xSp``
    # ``M = xSp``

    # p channel pSp = S * 0.5
    K1p = copy(K1S) * 0.5
    K2p = copy(K2S) * 0.5

    # t channel pSp = (D - M) * 0.5
    K1t = copy(K1D - K1M) * 0.5
    K2t = copy(K2D - K2M) * 0.5

    # a channel pSp = t channel xSp * (-1) = M * (-1)
    K1a = copy(K1M) * -1
    K2a = copy(K2M) * -1


    _mB = MatsubaraMesh(T, 1, Boson)
    _mF = MatsubaraMesh(T, 1, Fermion)
    _K3 = MeshFunction(_mB, _mF, _mF)
    set!(_K3, 0)

    γp = fdDGAsolver.Channel(K1p, K2p, _K3)
    γt = fdDGAsolver.Channel(K1t, K2t, _K3)
    γa = fdDGAsolver.Channel(K1a, K2a, _K3)

    Γ_K12 = Vertex(RefVertex(T, U, eltype(γa)), γp, γt, γa)

    # Subtract asymptotic contributions from the core
    for ind in eachindex(Ft_p.data)
        Ω, ν, ω = value.(to_meshes(Ft_p, ind))
        Ft_p[ind] = (F_D(Ω, ν, ω) - F_M(Ω, ν, ω)) * 0.5
        Ft_x[ind] = F_M(Ω, ν, ω)
        Fp_p[ind] = F_S(Ω, ν, ω) * 0.5
        Fp_x[ind] = -F_S(Ω, ν, ω) * 0.5
        Ft_p.data[ind] -= Γ_K12(Ω, ν, ω, tCh, pSp)
        Ft_x.data[ind] -= Γ_K12(Ω, ν, ω, tCh, xSp)
        Fp_p.data[ind] -= Γ_K12(Ω, ν, ω, pCh, pSp)
        Fp_x.data[ind] -= Γ_K12(Ω, ν, ω, pCh, xSp)
    end

    # Plot core vertex

    fig, plotaxes = subplots(4, 2, figsize=(6, 12); sharex=true, sharey=true)

    vmax = 0.3

    for (i, Λ) in enumerate([Ft_p, Ft_x, Fp_p, Fp_x])
        plotaxes[i, 1].imshow(real.(Λ.data[1,:,:]); vmin=-vmax, vmax, cmap="RdBu_r", aspect="auto", interpolation="none")
        plotaxes[i, 2].imshow(imag.(Λ.data[1,:,:]); vmin=-vmax, vmax, cmap="RdBu_r", aspect="auto", interpolation="none")
    end

    display(fig); close(fig)
end


begin
    # Plot K1 vertex
    for K1 in [K1D, K1M, K1S]
        plot(values(K1.meshes[1]), real.(K1.data))
    end
    xlim([-20, 20])
    fig = gcf(); display(fig); close(fig)
end

begin
    # Plot K2 vertex
    vmax = 0.1

    fig, plotaxes = subplots(3, 2, figsize=(6, 9); sharex=true, sharey=true)

    for (i, K2) in enumerate([K2D, K2M, K2S])
        plotaxes[i, 1].imshow(real.(K2.data); vmin=-vmax, vmax, cmap="RdBu_r", aspect="auto", interpolation="none")
        plotaxes[i, 2].imshow(imag.(K2.data); vmin=-vmax, vmax, cmap="RdBu_r", aspect="auto", interpolation="none")
    end

    for ax in plotaxes
        ax.set_xlim([40, 60])
        ax.set_ylim([60, 90])
    end

    display(fig); close(fig)
end

begin
    # Check the local Green fucntion G can be reproduced by the lattice Dyson equation
    T = 0.5
    μ = 0.
    t1 = -0.25
    U = 2.089
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    nmax = 8
    nG  = 12nmax
    nK1 = 8nmax
    nK2 = (nmax + 1, nmax)
    nK3 = (nmax + 1, nmax)
    mK_G = BrillouinZoneMesh(BrillouinZone(48, k1, k2))

    mG = MatsubaraMesh(T, nG, Fermion)
    Gbare = hubbard_bare_Green(mG, mK_G; μ, t1)
    G0 = copy(Gbare)
    Σ0 = copy(Gbare)
    set!(G0, 0)
    set!(Σ0, 0)
    for ν in MatsubaraFunctions.meshes(G0, Val(1))
        view(G0, ν, :) .= G[value(ν)]
        view(Σ0, ν, :) .= Σ[value(ν)]
    end
    G = copy(Gbare)
    Dyson!(G, Σ0, Gbare)

    y1 = [G0[ν, kSW] for ν in fdDGAsolver.meshes(G, 1)]
    y2 = [G[ν, kSW] for ν in fdDGAsolver.meshes(G, 1)]
    @info norm(y1 .- y2)
end
