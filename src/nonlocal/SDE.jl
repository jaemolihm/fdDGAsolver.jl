# Implementations for NL_ParquetSolver (nonlocal vertex with bosonic momentum dependence)

function SDE_channel_L_pp!(
    Lpp   :: NL_MF_K2{Q},
    Πpp   :: NL_MF_Π{Q},
    F     :: Union{Vertex{Q}, NL_Vertex{Q}},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :, P)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i])

            val += bare_vertex(F) * Πslice[i] * F.γp(Ω, Ω - ω, ν, P)
        end

        return temperature(F) * val
    end

    # compute Lpp
    SGpp2(Lpp, InitFunction{3, Q}(diagram); mode)

    return nothing
end

function SDE_channel_L_ph!(
    Lph   :: NL_MF_K2{Q},
    Πph   :: NL_MF_Π{Q},
    F     :: Union{Vertex{Q}, NL_Vertex{Q}},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :, P)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i])

            val += bare_vertex(F) * Πslice[i] * (F.γt(Ω, ν, ω, P) + F.γa(Ω, ν, ω, P))
        end

        return temperature(F) * val
    end

    # compute Lph
    SGph2(Lph, InitFunction{3, Q}(diagram); mode)

    return nothing
end

# U * Π * (F - U) for F <: RefVertex

function SDE_channel_L_pp!(
    Lpp   :: NL_MF_K2{Q},
    Πpp   :: NL_MF_Π{Q},
    F     :: RefVertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)

        for iω in eachindex(meshes(Πpp, Val(2)))
            ω = value(meshes(Πpp, Val(2))[iω])

            # RefVertex has no momentum dependence
            val += bare_vertex(F) * Πpp[Ω, ω, P] * (F(Ω, Ω - ω, ν, pCh, pSp) - bare_vertex(F, pCh, pSp))
        end

        return temperature(F) * val
    end

    # compute Lpp
    SGpp2(Lpp, InitFunction{3, Q}(diagram); mode)

    return nothing
end

function SDE_channel_L_ph!(
    Lph   :: NL_MF_K2{Q},
    Πph   :: NL_MF_Π{Q},
    F     :: RefVertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)

        for iω in eachindex(meshes(Πph, Val(2)))
            ω = value(meshes(Πph, Val(2))[iω])

            # RefVertex has no momentum dependence
            val += bare_vertex(F) * Πph[Ω, ω, P] * (F(Ω, ν, ω, aCh, pSp) + F(Ω, ν, ω, tCh, pSp) - bare_vertex(F, aCh, pSp) - bare_vertex(F, tCh, pSp))
        end

        return temperature(F) * val
    end

    # compute Lph
    SGph2(Lph, InitFunction{3, Q}(diagram); mode)

    return nothing
end


function SDE!(
    Σ     :: NL_MF_G{Q},
    G     :: NL_MF_G{Q},
    Πpp   :: NL_MF_Π{Q},
    Πph   :: NL_MF_Π{Q},
    Lpp   :: NL_MF_K2{Q},
    Lph   :: NL_MF_K2{Q},
    F     :: Union{NL_Vertex{Q}, Vertex{Q}, RefVertex{Q}},
    SGΣ   :: SymmetryGroup,
    SGpp2 :: SymmetryGroup,
    SGph2 :: SymmetryGroup,
    ;
    mode  :: Symbol,
    use_real_space :: Bool = true,
    include_U² = true,
    include_Hartree = true,
    )     :: NL_MF_G{Q} where {Q}
    # γa, γp, γt contribution to the self-energy in the asymptotic decomposition

    SDE_channel_L_pp!(Lpp, Πpp, F, SGpp2; mode)
    SDE_channel_L_ph!(Lph, Πph, F, SGph2; mode)

    meshK_G = meshes(G, Val(2))

    if use_real_space
        # Real-space evaluation
        # Σ( R) += G(-R) * Lpp(R)
        # Σ(-R) += G(-R) * Lph(R)

        LG = bz(meshes(G, Val(2))).L
        LΣ = bz(meshes(Σ, Val(2))).L
        L  = bz(meshes(Lpp, Val(3))).L

        n1 = length(meshes(Lpp, Val(1)))
        n2 = length(meshes(Lpp, Val(2)))

        G_data_R = fft(reshape(G.data, :, LG, LG), (2, 3)) / LG^2
        Σ_data_R = zeros(eltype(Σ.data), length(meshes(Σ, Val(1))), LΣ, LΣ)

        Lpp_data_R = fft(reshape(Lpp.data, n1, n2, L, L), (3, 4)) / L^2
        Lph_data_R = fft(reshape(Lph.data, n1, n2, L, L), (3, 4)) / L^2

        Rs_L_1d = (-div(L, 2)) : div(L, 2)
        Rs = collect(Iterators.product(Rs_L_1d, Rs_L_1d))

        for (R1, R2) in Rs
            R_vec = (R1, R2)

            weight = 1.0
            if mod(L, 2) == 0
                abs(R1) == div(L, 2) && (weight /= 2)
                abs(R2) == div(L, 2) && (weight /= 2)
            end

            # Index of R in L(R, Rp)
            iR_L  = mod.(R_vec, (L, L)) .+ 1

            # Index of -R in G
            imR_G = mod.(.-R_vec, (LG, LG)) .+ 1

            # Index of R and -R in Σ
            ipR_Σ = mod.(  R_vec, (LΣ, LΣ)) .+ 1
            imR_Σ = mod.(.-R_vec, (LΣ, LΣ)) .+ 1

            Lpp_R = MeshFunction((meshes(Lpp, Val(1)), meshes(Lpp, Val(2))), view(Lpp_data_R, :, :, iR_L...))
            Lph_R = MeshFunction((meshes(Lph, Val(1)), meshes(Lph, Val(2))), view(Lph_data_R, :, :, iR_L...))

            G_R = MeshFunction((meshes(G, Val(1)),), view(G_data_R, :, imR_G...))
            Σ_R_pp = MeshFunction((meshes(Σ, Val(1)),), view(Σ_data_R, :, ipR_Σ...))
            Σ_R_ph = MeshFunction((meshes(Σ, Val(1)),), view(Σ_data_R, :, imR_Σ...))

            for iν in eachindex(meshes(Lpp, Val(2)))
                ν = value(meshes(Lpp, Val(2))[iν])

                if is_inbounds(ν, meshes(Σ, Val(1)))

                    for iΩ in eachindex(meshes(Lpp, Val(1)))
                        Ω = value(meshes(Lpp, Val(1))[iΩ])

                        if is_inbounds(Ω - ν, meshes(G, Val(1)))
                            Σ_R_pp[ν] += G_R[Ω - ν] * Lpp_R[iΩ, iν] * weight
                            Σ_R_ph[ν] += G_R[Ω + ν] * Lph_R[iΩ, iν] * weight
                        end
                    end

                end
            end

        end

        Σ_data_R .*= temperature(F)

        Σ.data .= reshape(bfft(Σ_data_R, (2, 3)), :, LΣ^2)

        SGΣ(Σ)

    else
        # Momentum space evaluation
        # Σ(k) += G(P - k) * Lpp(P)
        # Σ(k) += G(P + k) * Lph(P)

        # model the diagram
        @inline function diagram(wtpl)

            ν, k = wtpl
            val = zero(Q)

            k_vec = euclidean(k, meshK_G)

            for iP in eachindex(meshes(Lpp, Val(3)))

                P = value(meshes(Lpp, Val(3))[iP])
                P_vec = euclidean(P, meshes(Lpp, Val(3)))

                Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec - k_vec, meshK_G)]
                Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec + k_vec, meshK_G)]

                if is_inbounds(ν, meshes(Lpp, Val(2)))
                    Lppslice = view(Lpp, :, ν, P)
                    Lphslice = view(Lph, :, ν, P)

                    for i in eachindex(Lppslice)
                        Ω = value(meshes(Lpp, Val(1))[i])

                        if is_inbounds(Ω - ν, meshes(G, Val(1)))
                            val += G[Ω - ν, Pmk_G] * Lppslice[i]
                        end

                        if is_inbounds(Ω + ν, meshes(G, Val(1)))
                            val += G[Ω + ν, Ppk_G] * Lphslice[i]
                        end
                    end
                end

            end

            return temperature(F) * val / length(meshes(Πpp, Val(3)))
        end

        # compute Σ
        SGΣ(Σ, InitFunction{2, Q}(diagram); mode)
    end


    if include_U²
        Σ_U² = SDE_U2_using_G(Σ, G, SGΣ, bare_vertex(F); mode)
        # Σ_U² = SDE_U2_using_Π(Σ, G, Πpp, Πph, SGΣ, bare_vertex(F); mode)
        add!(Σ, Σ_U²)
    end

    if include_Hartree
        n = compute_occupation(G)
        # We store im * Σ in S.Σ, so we multiply im.
        Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
    end

    return Σ
end


function SDE_U2_using_Π(
    Σ   :: NL_MF_G{Q},
    G   :: NL_MF_G{Q},
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    SGΣ :: SymmetryGroup,
    U   :: Number,
    ;
    mode :: Symbol,
    )   :: NL_MF_G{Q} where {Q}
    # 2nd-order perturbative contribution to the self-energy

    T = temperature(meshes(Σ, Val(1)))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    # Integrate fermionic frequency and momentum for the bubble

    Πppsum = MeshFunction((meshes(Πpp, Val(1)), meshes(Πpp, Val(3)),),
        dropdims(sum(Πpp.data, dims=(2,4)), dims=(2,4)) .* (T / length(meshes(Πpp, Val(4)))))
    Πphsum = MeshFunction((meshes(Πph, Val(1)), meshes(Πph, Val(3)),),
        dropdims(sum(Πph.data, dims=(2,4)), dims=(2,4)) .* (T / length(meshes(Πph, Val(4)))))

    meshK_G = meshes(G, Val(2))

    # model the diagram
    @inline function diagram(wtpl)

        ν, k = wtpl
        val  = zero(Q)

        k_vec = euclidean(k, meshK_G)

        for iP in eachindex(meshes(Πppsum, Val(2)))

            P = value(meshes(Πppsum, Val(2))[iP])
            P_vec = euclidean(P, meshes(Πppsum, Val(2)))

            Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec - k_vec, meshK_G)]
            Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec + k_vec, meshK_G)]

            for iΩ in eachindex(meshes(Πppsum, Val(1)))

                Ω = value(meshes(Πppsum, Val(1))[iΩ])

                if is_inbounds(Ω - ν, meshes(G, Val(1)))
                    val += G[Ω - ν, Pmk_G] * Πppsum[iΩ, iP]
                end

                if is_inbounds(Ω + ν, meshes(G, Val(1)))
                    val += G[Ω + ν, Ppk_G] * Πphsum[iΩ, iP]
                end

            end

        end

        return val * U^2 * T / 2 / length(meshes(Πph, Val(3)))
    end

    # compute Σ
    SGΣ(Σ_U², InitFunction{2, Q}(diagram); mode)

    return Σ_U²
end


function SDE_U2_using_G(
    Σ   :: NL_MF_G{Q},
    G   :: NL_MF_G{Q},
    SGΣ :: SymmetryGroup,
    U   :: Number,
    ;
    mode  :: Symbol,
    )   :: NL_MF_G{Q} where {Q}

    # Σ_U²(ν, R) = U² * T² * ∑_{ω1, ω2, ν} G(ω1, -R) * G(ω2, R) * G(ω1 - ω2 + ν, R)

    T = temperature(meshes(Σ, Val(1)))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    L = bz(meshes(G, Val(2))).L
    G_pR =  fft(reshape(G.data, :, L, L), (2, 3)) / L^2
    G_mR = bfft(reshape(G.data, :, L, L), (2, 3)) / L^2

    Σ_U²_R = zeros(eltype(Σ.data), length(meshes(Σ, Val(1))), L, L)

    Threads.@threads for (iR1, iR2) in collect(Iterators.product(axes(G_pR, 2), axes(G_pR, 3)))
        Gp = MeshFunction((meshes(G, Val(1)),), view(G_pR, :, iR1, iR2))
        Gm = MeshFunction((meshes(G, Val(1)),), view(G_mR, :, iR1, iR2))

        for i2 in eachindex(meshes(G, Val(1))), i1 in eachindex(meshes(G, Val(1)))
            ω1 = value(meshes(G, Val(1))[i1])
            ω2 = value(meshes(G, Val(1))[i2])
            gg = Gm[ω1] * Gp[ω2]

            for iν in eachindex(meshes(Σ, Val(1)))
                ν = value(meshes(Σ, Val(1))[iν])

                if is_inbounds(ω1 - ω2 + ν, meshes(G, Val(1)))
                    Σ_U²_R[iν, iR1, iR2] += gg * Gp[ω1 - ω2 + ν]
                end
            end
        end
    end

    Σ_U²_R .*= U^2 * T^2

    Σ_U².data .= reshape(bfft(Σ_U²_R, (2, 3)), :, L^2)

    SGΣ(Σ_U²)

    Σ_U²
end;


function self_energy_sanity_check(Σ :: NL_MF_G)
    passed = true
    # sanity check
    for k in meshes(Σ, Val(2)), ν in meshes(Σ, Val(1))
        if plain_value(ν) > 0 && real(Σ[ν, k]) < 0
            passed = false
            @warn "Σ violates causality at (n, k) = $(index(ν)), $(index(k))"
        elseif plain_value(ν) < 0 && real(Σ[ν, k]) > 0
            passed = false
            @warn "Σ violates causality at (n, k) = $(index(ν)), $(index(k))"
        end
    end
    passed
end
