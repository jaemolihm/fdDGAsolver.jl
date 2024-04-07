# Implementations for NL_ParquetSolver (nonlocal vertex with bosonic momentum dependence)

function SDE_channel_L_pp(
    Πpp   :: NL_MF_Π{Q},
    F     :: NL_Vertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: NL_MF_K2{Q} where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, 2)[i.I[1]])
            q = value(meshes(Πpp, 4)[i.I[2]])

            val += bare_vertex(F) * Πslice[i] * F.γp(Ω, Ω - ω, ν, P, P - q, k0)
        end

        return temperature(F) * val / length(meshes(Πpp, 4))
    end

    # compute Lpp
    Lpp = copy(F.γp.K2)

    SGpp2(Lpp, InitFunction{3, Q}(diagram); mode = mode)

    return Lpp
end

function SDE_channel_L_ph(
    Πph   :: NL_MF_Π{Q},
    F     :: NL_Vertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: NL_MF_K2{Q} where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, 2)[i.I[1]])
            q = value(meshes(Πph, 4)[i.I[2]])

            val += bare_vertex(F) * Πslice[i] * (F.γt(Ω, ν, ω, P, k0, q) + F.γa(Ω, ν, ω, P, k0, q))
        end

        return temperature(F) * val / length(meshes(Πph, 4))
    end

    # compute Lph
    Lph = copy(F.γt.K2)

    SGph2(Lph, InitFunction{3, Q}(diagram); mode)

    return Lph
end


function SDE!(
    Σ     :: NL_MF_G{Q},
    G     :: NL_MF_G{Q},
    Πpp   :: NL_MF_Π{Q},
    Πph   :: NL_MF_Π{Q},
    F     :: Union{NL_Vertex{Q}, Vertex{Q}, RefVertex{Q}},
    SGΣ   :: SymmetryGroup,
    SGpp2 :: SymmetryGroup,
    SGph2 :: SymmetryGroup,
    ;
    mode  :: Symbol,
    include_U² = true,
    include_Hartree = true,
    )     :: NL_MF_G{Q} where {Q}
    # γa, γp, γt contribution to the self-energy in the asymptotic decomposition

    Lpp = SDE_channel_L_pp(Πpp, F, SGpp2; mode)
    Lph = SDE_channel_L_ph(Πph, F, SGph2; mode)

    meshK_G = meshes(G, 2)

    # model the diagram
    @inline function diagram(wtpl)

        ν, k = wtpl
        val = zero(Q)

        k_vec = euclidean(k, meshK_G)

        for iP in eachindex(meshes(Lpp, 3))

            P = value(meshes(Lpp, 3)[iP])
            P_vec = euclidean(P, meshes(Lpp, 3))

            Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec - k_vec, meshK_G)]
            Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec + k_vec, meshK_G)]

            if is_inbounds(ν, meshes(Lpp, 2))
                Lppslice = view(Lpp, :, ν, P)
                Lphslice = view(Lph, :, ν, P)

                for i in eachindex(Lppslice)
                    Ω = value(meshes(Lpp, 1)[i])

                    if is_inbounds(Ω - ν, meshes(G, 1))
                        val += G[Ω - ν, Pmk_G] * Lppslice[i]
                    end

                    if is_inbounds(Ω + ν, meshes(G, 1))
                        val += G[Ω + ν, Ppk_G] * Lphslice[i]
                    end
                end
            end

        end

        return temperature(F) * val / length(meshes(Πpp, 3))
    end

    # compute Σ
    SGΣ(Σ, InitFunction{2, Q}(diagram); mode)

    if include_U²
        Σ_U² = SDE_U2(Σ, G, Πpp, Πph, SGΣ, bare_vertex(F); mode)
        add!(Σ, Σ_U²)
    end

    if include_Hartree
        n = compute_occupation(G)
        # We store im * Σ in S.Σ, so we multiply im.
        Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
    end

    return Σ
end


function SDE_U2(
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

    T = temperature(meshes(Σ, 1))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    # Integrate fermionic frequency and momentum for the bubble

    Πppsum = MeshFunction((meshes(Πpp, 1), meshes(Πpp, 3),),
        dropdims(sum(Πpp.data, dims=(2,4)), dims=(2,4)) .* (T / length(meshes(Πpp, 4))))
    Πphsum = MeshFunction((meshes(Πph, 1), meshes(Πph, 3),),
        dropdims(sum(Πph.data, dims=(2,4)), dims=(2,4)) .* (T / length(meshes(Πph, 4))))

    meshK_G = meshes(G, 2)

    # model the diagram
    @inline function diagram(wtpl)

        ν, k = wtpl
        val  = zero(Q)

        k_vec = euclidean(k, meshK_G)

        for iP in eachindex(meshes(Πppsum, 2))

            P = value(meshes(Πppsum, 2)[iP])
            P_vec = euclidean(P, meshes(Πppsum, 2))

            Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec - k_vec, meshK_G)]
            Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec + k_vec, meshK_G)]

            for iΩ in eachindex(meshes(Πppsum, 1))

                Ω = value(meshes(Πppsum, 1)[iΩ])

                if is_inbounds(Ω - ν, meshes(G, 1))
                    val += G[Ω - ν, Pmk_G] * Πppsum[iΩ, iP]
                end

                if is_inbounds(Ω + ν, meshes(G, 1))
                    val += G[Ω + ν, Ppk_G] * Πphsum[iΩ, iP]
                end

            end

        end

        return val * U^2 * T / 2 / length(meshes(Πph, 3))
    end

    # compute Σ
    SGΣ(Σ_U², InitFunction{2, Q}(diagram); mode)

    return Σ_U²
end


# function SDE_using_K12!(
#     Σ :: MF_G{Q},
#     G :: MF_G{Q},
#     F :: Vertex{Q},
#     SGΣ :: SymmetryGroup,
#     ;
#     mode :: Symbol,
#     include_Hartree = true,
#     ) :: MF_G{Q} where {Q}

#     # model the diagram
#     @inline function diagram(wtpl)

#         ν   = wtpl[1]
#         val = zero(Q)

#         for iω in eachindex(meshes(G, 1))
#             ω = value(meshes(G, 1)[iω])
#             # SDE using only K1 + K2 in p channel
#             val += G[ω] * (box_eval(F.γp.K1, ν + ω) + box_eval(F.γp.K2, ν + ω, ν))
#         end

#         return temperature(F) * val
#     end

#     # compute Σ
#     SGΣ(Σ, InitFunction{1, Q}(diagram); mode)

#     if include_Hartree
#         n = compute_occupation(G)
#         # We store im * Σ in S.Σ, so we multiply im.
#         Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
#     end

#     return Σ
# end

function self_energy_sanity_check(Σ :: NL_MF_G)
    passed = true
    # sanity check
    for k in meshes(Σ, 2), ν in meshes(Σ, 1)
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

function compute_occupation(G :: NL_MF_G)
    return 0.5 + imag(sum(G.data)) * temperature(meshes(G, 1)) / length(meshes(G, 2))
end
