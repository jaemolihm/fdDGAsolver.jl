# Implementations for NL2_ParquetSolver (nonlocal vertex with bosonic and fermionic momentum dependences)

function SDE_channel_L_pp(
    Πpp   :: NL_MF_Π{Q},
    F     :: NL2_Vertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: NL2_MF_K2{Q} where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i.I[1]])
            q = value(meshes(Πpp, Val(4))[i.I[2]])

            val += bare_vertex(F) * Πslice[i] * F.γp(Ω, Ω - ω, ν, P, P - q, k)
        end

        return temperature(F) * val / length(meshes(Πpp, Val(4)))
    end

    # compute Lpp
    Lpp = copy(F.γp.K2)

    SGpp2(Lpp, InitFunction{4, Q}(diagram); mode = mode)

    return Lpp
end

function SDE_channel_L_ph(
    Πph   :: NL_MF_Π{Q},
    F     :: NL2_Vertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: NL2_MF_K2{Q} where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i.I[1]])
            q = value(meshes(Πph, Val(4))[i.I[2]])

            val += bare_vertex(F) * Πslice[i] * (F.γt(Ω, ν, ω, P, k, q) + F.γa(Ω, ν, ω, P, k, q))
        end

        return temperature(F) * val / length(meshes(Πph, Val(4)))
    end

    # compute Lph
    Lph = copy(F.γt.K2)

    SGph2(Lph, InitFunction{4, Q}(diagram); mode)

    return Lph
end


function SDE!(
    Σ     :: NL_MF_G{Q},
    G     :: NL_MF_G{Q},
    Πpp   :: NL_MF_Π{Q},
    Πph   :: NL_MF_Π{Q},
    F     :: NL2_Vertex{Q},
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

    mG = meshes(G, Val(2))
    @assert mG == meshes(Lpp, Val(4)) == meshes(Lph, Val(4))

    # model the diagram
    @inline function diagram(wtpl)

        ν, k = wtpl
        val = zero(Q)

        for iP in eachindex(meshes(Lpp, Val(3)))

            P = value(meshes(Lpp, Val(3))[iP])

            if is_inbounds(ν, meshes(Lpp, Val(2)))
                Lppslice = view(Lpp, :, ν, P, k)
                Lphslice = view(Lph, :, ν, P, k)

                for i in eachindex(Lppslice)
                    Ω = value(meshes(Lpp, Val(1))[i])

                    if is_inbounds(Ω - ν, meshes(G, Val(1)))
                        val += G[Ω - ν, fold_back(P - k, mG)] * Lppslice[i]
                    end

                    if is_inbounds(Ω + ν, meshes(G, Val(1)))
                        val += G[Ω + ν, fold_back(P - k, mG)] * Lphslice[i]
                    end
                end
            end

        end

        return temperature(F) * val / length(meshes(Πpp, Val(3)))
    end

    # compute Σ
    SGΣ(Σ, InitFunction{2, Q}(diagram); mode)

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
