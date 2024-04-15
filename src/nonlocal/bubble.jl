function bubbles_momentum_space!(S :: AbstractSolver)
    bubbles_momentum_space!(S.Πpp, S.Πph, S.G)
end


function bubbles_real_space!(S :: AbstractSolver)
    bubbles_real_space!(S.Πpp, S.Πph, S.G)
end

function bubbles_momentum_space!(
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    G   :: NL_MF_G{Q},
    ) :: Nothing where {Q}

    set!(Πpp, 0)
    set!(Πph, 0)
    meshK_G = meshes(G, Val(2))

    for iP in eachindex(meshes(Πpp, Val(3)))
        P = euclidean(meshes(Πpp, Val(3))[iP], meshes(Πpp, Val(3)))

        for ik in eachindex(meshes(G, Val(2)))
            k = euclidean(meshes(G, Val(2))[ik], meshes(G, Val(2)))

            k_G   = meshK_G[MatsubaraFunctions.mesh_index(    k, meshK_G)]
            Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P - k, meshK_G)]
            Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P + k, meshK_G)]

            for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
                Ω = value(meshes(Πpp, Val(1))[iΩ])
                ν = value(meshes(Πpp, Val(2))[iν])

                if is_inbounds(ν, meshes(G, Val(1)))
                    if is_inbounds(Ω - ν, meshes(G, Val(1)))
                        Πpp[iΩ, iν, iP] += G[ν, k_G] * G[Ω - ν, Pmk_G]
                    end

                    if is_inbounds(Ω + ν, meshes(G, Val(1)))
                        Πph[iΩ, iν, iP] += G[ν, k_G] * G[Ω + ν, Ppk_G]
                    end
                end
            end
        end

    end

    Πpp.data ./= length(meshes(G, Val(2)))
    Πph.data ./= length(meshes(G, Val(2)))

    return nothing
end




function bubbles_real_space!(
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    G   :: NL_MF_G{Q},
    ) :: Nothing where {Q}

    # G(R) = ∑_k G(k) exp(-ikR) / N_k
    # Πpp(R) = G(R) * G( R)
    # Πph(R) = G(R) * G(-R)
    # Π(P) = ∑_{R} Π(R) exp(iPR)

    set!(Πpp, 0)
    set!(Πph, 0)

    LG = bz(meshes(G, Val(2))).L
    L  = bz(meshes(Πpp, Val(3))).L

    n1 = length(meshes(Πpp, Val(1)))
    n2 = length(meshes(Πpp, Val(2)))

    G_real_space = fft(reshape(G.data, :, LG, LG), (2, 3)) / LG^2

    Πpp_R = Base.ReshapedArray(Πpp.data, (n1, n2, L, L), ())
    Πph_R = Base.ReshapedArray(Πph.data, (n1, n2, L, L), ())

    # Scan G(R) and G(R') for R and R' in [-L/2, L/2]²
    Rs_G_1d = (-div(L, 2)) : div(L, 2)
    Rs = collect(Iterators.product(Rs_G_1d, Rs_G_1d))

    for (R1, R2) in Rs
        R_vec = (R1, R2)

        weight = 1.0
        if mod(LG, 2) == 0
            abs(R1) == div(LG, 2) && (weight /= 2)
            abs(R2) == div(LG, 2) && (weight /= 2)
        end

        # Index of R and Rp in G
        ipR_G = mod.(  R_vec, (LG, LG)) .+ 1
        imR_G = mod.(.-R_vec, (LG, LG)) .+ 1

        # Index of R, Rp - R, and Rp + R in Π
        iR_Π   = mod.(R_vec, (L, L)) .+ 1

        G_pR = MeshFunction((meshes(G, Val(1)),), view(G_real_space, :, ipR_G...))
        G_mR = MeshFunction((meshes(G, Val(1)),), view(G_real_space, :, imR_G...))

        for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
            Ω = value(meshes(Πpp, Val(1))[iΩ])
            ν = value(meshes(Πpp, Val(2))[iν])

            Πpp_R[iΩ, iν, iR_Π...] += G_pR(Ω - ν) * G_pR(ν) * weight
            Πph_R[iΩ, iν, iR_Π...] += G_pR(Ω + ν) * G_mR(ν) * weight
        end
    end

    bfft!(Πpp_R, (3, 4))
    bfft!(Πph_R, (3, 4))

    return nothing
end
