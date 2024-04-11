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
    use_real_space :: Bool = true,
    include_U² = true,
    include_Hartree = true,
    )     :: NL_MF_G{Q} where {Q}
    # γa, γp, γt contribution to the self-energy in the asymptotic decomposition

    Lpp = SDE_channel_L_pp(Πpp, F, SGpp2; mode)
    Lph = SDE_channel_L_ph(Πph, F, SGph2; mode)

    if use_real_space
        # Real-space evaluation
        # Σ( R + R') += G(-R) * Lpp(R, R')
        # Σ(-R + R') += G(-R) * Lph(R, R')

        LG = bz(meshes(G, 2)).L
        LΣ = bz(meshes(Σ, 2)).L
        L  = bz(meshes(Lpp, 3)).L

        n1 = length(meshes(Lpp, 1))
        n2 = length(meshes(Lpp, 2))

        G_data_R = fft(reshape(G.data, :, LG, LG), (2, 3)) / LG^2
        Σ_data_R = zeros(eltype(Σ.data), length(meshes(Σ, 1)), LΣ, LΣ)

        Lpp_data_R = fft(reshape(Lpp.data, n1, n2, L, L, L, L), (3, 4, 5, 6)) / L^4
        Lph_data_R = fft(reshape(Lph.data, n1, n2, L, L, L, L), (3, 4, 5, 6)) / L^4

        Rs_L_1d = (-div(L, 2)) : div(L, 2)
        Rs = collect(Iterators.product(Rs_L_1d, Rs_L_1d))

        for (Rp1, Rp2) in Rs
            Rp_vec = (Rp1, Rp2)

            # Index of Rp in L(R, Rp)
            iRp_L = mod.(Rp_vec, (L, L)) .+ 1

            for (R1, R2) in Rs
                R_vec = (R1, R2)

                weight = 1.0
                if mod(L, 2) == 0
                    abs(Rp1) == div(L, 2) && (weight /= 2)
                    abs(Rp2) == div(L, 2) && (weight /= 2)
                    abs(R1) == div(L, 2) && (weight /= 2)
                    abs(R2) == div(L, 2) && (weight /= 2)
                end

                # Index of R in L(R, Rp)
                iR_L  = mod.(R_vec, (L, L)) .+ 1

                # Index of -R in G
                imR_G = mod.(.-R_vec, (LG, LG)) .+ 1

                # Index of R + Rp and -R + Rp in Σ
                iRpR_Σ = mod.(   R_vec .+ Rp_vec, (LΣ, LΣ)) .+ 1
                iRmR_Σ = mod.(.- R_vec .+ Rp_vec, (LΣ, LΣ)) .+ 1

                Lpp_R = MeshFunction((meshes(Lpp, 1), meshes(Lpp, 2)), view(Lpp_data_R, :, :, iR_L..., iRp_L...))
                Lph_R = MeshFunction((meshes(Lph, 1), meshes(Lph, 2)), view(Lph_data_R, :, :, iR_L..., iRp_L...))

                G_R = MeshFunction((meshes(G, 1),), view(G_data_R, :, imR_G...))
                Σ_R_pp = MeshFunction((meshes(Σ, 1),), view(Σ_data_R, :, iRpR_Σ...))
                Σ_R_ph = MeshFunction((meshes(Σ, 1),), view(Σ_data_R, :, iRmR_Σ...))

                for iν in eachindex(meshes(Lpp, 2))
                    ν = value(meshes(Lpp, 2)[iν])

                    if is_inbounds(ν, meshes(Σ, 1))

                        for iΩ in eachindex(meshes(Lpp, 1))
                            Ω = value(meshes(Lpp, 1)[iΩ])

                            if is_inbounds(Ω - ν, meshes(G, 1))
                                Σ_R_pp[ν] += G_R[Ω - ν] * Lpp_R[iΩ, iν] * weight
                                Σ_R_ph[ν] += G_R[Ω + ν] * Lph_R[iΩ, iν] * weight
                            end
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
        # Σ(k) += G(P - k) * Lpp(P, k)
        # Σ(k) += G(P + k) * Lph(P, k)

        # It is required that Σ and L has the same mesh.
        # (One may use linear interpolation, but that is not implemented here.)
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
                            val += G[Ω + ν, fold_back(P + k, mG)] * Lphslice[i]
                        end
                    end
                end

            end

            return temperature(F) * val / length(meshes(Lpp, 3))
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
