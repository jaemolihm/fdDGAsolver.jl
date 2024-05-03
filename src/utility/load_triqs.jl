# Load a local Vertex from TRIQS data files.


# Spin convention used in the TRIQS input
# ``S = pSp - xSp``
# ``T = pSp + xSp``
# ``D = 2 * pSp + xSp``
# ``M = xSp``

function _drop_first_dim(f)
    MeshFunction(f.meshes[2:end], dropdims(f.data, dims=1))
end

function parse_triqs_data(prefix, T, U; params, half_filling = false, symmetrize = true, filename_output = nothing)

    filename_G = "$(prefix)_DCA.h5"
    filename_chi = "$(prefix)_chi3_fromDCA.h5"
    filename_F = "$(prefix)_F_fromDCA.h5"
    Ω0 = MatsubaraFrequency(T, 0, Boson)

    #-----------------------------------------------------------------------------#
    # Green function and self-energy

    f = h5open(filename_G, "r")
    occ = h5read(filename_G, "density")
    G0 = _drop_first_dim(load_triqs_gf(f, "cG0_k_iw"));
    G = _drop_first_dim(load_triqs_gf(f, "cG_k_iw"));
    Σ = _drop_first_dim(load_triqs_gf(f, "cSigma_k_iw"));
    close(f)

    if half_filling && abs(occ - 0.5) > 1e-5
        error("half_filling is set to true but the occupation is not 0.5.")
    end

    # Subtract U/2 because we put Σ = 0 at half filling and in TRIQS Σ = U/2 is half filling.
    Σ.data .-= U/2

    # We factor out -im from G and Σ and store im * G and im * Σ
    mult!(G0, im)
    mult!(G, im)
    mult!(Σ, im)


    # Update G0 to take the chemical potential shift by U/2 into account
    Σ_const = copy(G0)
    G0_new = copy(G0)
    set!(Σ_const, U/2 * im)
    Dyson!(G0_new, Σ_const, G0)
    G0 = G0_new


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

    # Convert from the D, M, S spin basis to the pSp spin basis in each channel.
    # p channel pSp = S * 0.5
    K1p = copy(K1S) * 0.5
    K2p = copy(K2S) * 0.5

    # t channel pSp = (D - M) * 0.5
    K1t = copy(K1D - K1M) * 0.5
    K2t = copy(K2D - K2M) * 0.5

    # a channel pSp = t channel xSp * (-1) = M * (-1)
    K1a = copy(K1M) * -1
    K2a = copy(K2M) * -1

    # Add dummy K3 vertex (we do not decompose the core part into the K3 vertex and the
    # 2-particle irreducible part, so we set K3 = 0, and the rest is in the RefVertex.)
    _mB = MatsubaraMesh(T, 1, Boson)
    _mF = MatsubaraMesh(T, 1, Fermion)
    _K3 = MeshFunction(_mB, _mF, _mF)
    set!(_K3, 0)

    γp = fdDGAsolver.Channel(K1p, K2p, _K3)
    γt = fdDGAsolver.Channel(K1t, K2t, _K3)
    γa = fdDGAsolver.Channel(K1a, K2a, _K3)

    # Define a K12-only vertex, used to subtract the asymptotic contribution from the core.
    Γ_K12 = Vertex(RefVertex(T, U, eltype(γa)), γp, γt, γa)

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

    # Subtract asymptotic contributions from the core
    for ind in eachindex(Ft_p.data)
        Ω, ν, ω = value.(to_meshes(Ft_p, ind))
        # t channel
        # F_D = (tCh, D), F_M = (tCh, M), F_S = (pCh, S)
        # tCh, pSp = [(tCh, D) - (tCh, M)] / 2
        # tCh, xSp = (tCh, M)
        Ft_p[ind] = (F_D(Ω, ν, ω) - F_M(Ω, ν, ω)) * 0.5
        Ft_x[ind] = F_M(Ω, ν, ω)
        Ft_p.data[ind] -= Γ_K12(Ω, ν, ω, tCh, pSp)
        Ft_x.data[ind] -= Γ_K12(Ω, ν, ω, tCh, xSp)

        # p Channel
        # pCh, pSp = [(tCh, D) - (tCh, M)] / 2 at convert_frequency(pCh -> tCh)
        # pCh, xSp = (tCh, M) at convert_frequency(pCh -> tCh)
        Fp_p[ind] = (F_D(convert_frequency(Ω, ν, ω, pCh, tCh)...) - F_M(convert_frequency(Ω, ν, ω, pCh, tCh)...)) / 2
        Fp_x[ind] = F_M(convert_frequency(Ω, ν, ω, pCh, tCh)...)
        if all(is_inbounds.(convert_frequency(Ω, ν, ω, pCh, tCh), F_D.meshes))
            # Since we do frequency conversion, the box for Fp_p and Fp_x are different from
            # the box for F_D and F_M. We need to subtract the asymptotic contribution only
            # from the box of F_D and F_M.
            Fp_p.data[ind] -= Γ_K12(Ω, ν, ω, pCh, pSp)
            Fp_x.data[ind] -= Γ_K12(Ω, ν, ω, pCh, xSp)
        end
    end

    Λ = RefVertex(U, Fp_p, Fp_x, Ft_p, Ft_x)

    Γ = Vertex(Λ, γp, γt, γa)

    if half_filling
        # Impose half filling symmetry.
        # The physical Green function and self-energy are purely imaginary.
        # Since we factor out -im from them,, G and Σ are purely real.
        # The vertices are purely real.
        println("Half-filling violation")
        println("G0    : ", maximum(abs.(imag.(G0.data))))
        println("G     : ", maximum(abs.(imag.(G.data))))
        println("Σ     : ", maximum(abs.(imag.(Σ.data))))
        println("γp.K1 : ", maximum(abs.(imag.(Γ.γp.K1.data))))
        println("γt.K1 : ", maximum(abs.(imag.(Γ.γt.K1.data))))
        println("γa.K1 : ", maximum(abs.(imag.(Γ.γa.K1.data))))
        println("γp.K2 : ", maximum(abs.(imag.(Γ.γp.K2.data))))
        println("γt.K2 : ", maximum(abs.(imag.(Γ.γt.K2.data))))
        println("γa.K2 : ", maximum(abs.(imag.(Γ.γa.K2.data))))
        println("Fp_p  : ", maximum(abs.(imag.(Γ.F0.Fp_p.data))))
        println("Fp_x  : ", maximum(abs.(imag.(Γ.F0.Fp_x.data))))
        println("Ft_p  : ", maximum(abs.(imag.(Γ.F0.Ft_p.data))))
        println("Ft_x  : ", maximum(abs.(imag.(Γ.F0.Ft_x.data))))

        println("Imposing half-filling...")
        G0.data .= real.(G0.data)
        G.data  .= real.(G.data)
        Σ.data  .= real.(Σ.data)
        Γ.γp.K1.data .= real.(Γ.γp.K1.data)
        Γ.γt.K1.data .= real.(Γ.γt.K1.data)
        Γ.γa.K1.data .= real.(Γ.γa.K1.data)
        Γ.γp.K2.data .= real.(Γ.γp.K2.data)
        Γ.γt.K2.data .= real.(Γ.γt.K2.data)
        Γ.γa.K2.data .= real.(Γ.γa.K2.data)
        Γ.F0.Fp_p.data .= real.(Γ.F0.Fp_p.data)
        Γ.F0.Fp_x.data .= real.(Γ.F0.Fp_x.data)
        Γ.F0.Ft_p.data .= real.(Γ.F0.Ft_p.data)
        Γ.F0.Ft_x.data .= real.(Γ.F0.Ft_x.data)
    end

    if symmetrize
        println("Imposing symmetries...")

        # particle-particle channel
        SGpp1 = my_SymmetryGroup([Symmetry{1}(sK1pp)], Γ.γp.K1)
        SGpp2 = my_SymmetryGroup([Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)], Γ.γp.K2)
        SGpp3 = my_SymmetryGroup([Symmetry{3}(sK3pp1), Symmetry{3}(sK3pp2), Symmetry{3}(sK3pp3)], Γ.F0.Fp_p)

        # particle-hole channels
        SGph1 = my_SymmetryGroup([Symmetry{1}(sK1ph)], Γ.γt.K1)
        SGph2 = my_SymmetryGroup([Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)], Γ.γt.K2)
        SGph3 = my_SymmetryGroup([Symmetry{3}(sK3ph1), Symmetry{3}(sK3ph2), Symmetry{3}(sK3ph3)], Γ.F0.Ft_p)

        println("symmetrization errors :")
        @info SGpp1(Γ.γp.K1)
        @info SGpp2(Γ.γp.K2)
        @info SGph1(Γ.γt.K1)
        @info SGph1(Γ.γa.K1)
        @info SGph2(Γ.γt.K2)
        @info SGph2(Γ.γa.K2)
        @info SGpp3(Γ.F0.Fp_p)
        @info SGpp3(Γ.F0.Fp_x)
        @info SGph3(Γ.F0.Ft_p)
        @info SGph3(Γ.F0.Ft_x)
    end

    if filename_output !== nothing
        f = h5open(filename_output, "w")
        save!(f, "G", G)
        save!(f, "G0", G0)
        save!(f, "Σ", Σ)
        save!(f, "Γ", Γ)
        f["occ"] = occ
        f["params/T"] = T
        f["params/U"] = U
        for key in keys(params)
            f["params/$key"] = getproperty(params, key)
        end
        close(f)
    end

    return (; G0, G, Σ, Γ, occ)
end


function load_triqs_data(filename)
    f = h5open(filename, "r")
    G = load_mesh_function(f, "G")
    G0 = load_mesh_function(f, "G0")
    Σ = load_mesh_function(f, "Σ")
    Γ = fdDGAsolver.load_vertex(Vertex, f, "Γ")
    occ = read(f, "occ")
    params = (; (Symbol(key) => val for (key, val) in read(f, "params"))...)
    close(f)
    (; G, G0, Σ, Γ, occ, params)
end
