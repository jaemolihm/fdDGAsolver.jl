function bubbles_momentum_space!(
    Πpp :: NL2_MF_Π{Q},
    Πph :: NL2_MF_Π{Q},
    G   :: NL_MF_G{Q},
    ) :: Nothing where {Q}

    set!(Πpp, 0)
    set!(Πph, 0)
    meshK_G = meshes(G, Val(2))

    for iP in eachindex(meshes(Πpp, Val(3))), ik in eachindex(meshes(Πpp, Val(4)))
        P = euclidean(meshes(Πpp, Val(3))[iP], meshes(Πpp, Val(3)))
        k = euclidean(meshes(Πpp, Val(4))[ik], meshes(Πpp, Val(4)))

        k_G   = meshK_G[MatsubaraFunctions.mesh_index(    k, meshK_G)]
        Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P - k, meshK_G)]
        Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P + k, meshK_G)]

        for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
            Ω = value(meshes(Πpp, Val(1))[iΩ])
            ν = value(meshes(Πpp, Val(2))[iν])

            if is_inbounds(ν, meshes(G, Val(1)))
                if is_inbounds(Ω - ν, meshes(G, Val(1)))
                    Πpp[iΩ, iν, iP, ik] = G[ν, k_G] * G[Ω - ν, Pmk_G]
                end

                if is_inbounds(Ω + ν, meshes(G, Val(1)))
                    Πph[iΩ, iν, iP, ik] = G[ν, k_G] * G[Ω + ν, Ppk_G]
                end
            end
        end

    end

    return nothing
end




function bubbles_real_space!(
    Πpp :: NL2_MF_Π{Q},
    Πph :: NL2_MF_Π{Q},
    G   :: NL_MF_G{Q},
    ) :: Nothing where {Q}

    # G(R) = ∑_k G(k) exp(-ikR) / N_k
    # Πpp(R, R' - R) = G(R) * G(R')
    # Πph(R, R' + R) = G(R) * G(R')
    # Π(P, k) = ∑_{R, R'} Π(R, R') exp(iPR) exp(ikR')

    set!(Πpp, 0)
    set!(Πph, 0)

    LG = bz(meshes(G, Val(2))).L
    L  = bz(meshes(Πpp, Val(3))).L

    n1 = length(meshes(Πpp, Val(1)))
    n2 = length(meshes(Πpp, Val(2)))

    G_real_space = fft(reshape(G.data, :, LG, LG), (2, 3)) / LG^2

    Πpp_R = Base.ReshapedArray(Πpp.data, (n1, n2, L, L, L, L), ())
    Πph_R = Base.ReshapedArray(Πph.data, (n1, n2, L, L, L, L), ())

    # Scan G(R) and G(R') for R and R' in [-L/2, L/2]²
    Rs_G_1d = (-div(L, 2)) : div(L, 2)
    Rs = collect(Iterators.product(Rs_G_1d, Rs_G_1d))

    for (Rp1, Rp2) in Rs
        Rp_vec = (Rp1, Rp2)

        for (R1, R2) in Rs
            R_vec = (R1, R2)

            weight = 1.0
            if mod(LG, 2) == 0
                abs(Rp1) == div(LG, 2) && (weight /= 2)
                abs(Rp2) == div(LG, 2) && (weight /= 2)
                abs(R1) == div(LG, 2) && (weight /= 2)
                abs(R2) == div(LG, 2) && (weight /= 2)
            end

            # Index of R and Rp in G
            iR_G  = mod.(R_vec,  (LG, LG)) .+ 1
            iRp_G = mod.(Rp_vec, (LG, LG)) .+ 1

            # Index of R, Rp - R, and Rp + R in Π
            iR_Π   = mod.(R_vec, (L, L)) .+ 1
            iRmR_Π = mod.(Rp_vec .- R_vec, (L, L)) .+ 1
            iRpR_Π = mod.(Rp_vec .+ R_vec, (L, L)) .+ 1

            G_R  = MeshFunction((meshes(G, Val(1)),), view(G_real_space, :, iR_G...))
            G_Rp = MeshFunction((meshes(G, Val(1)),), view(G_real_space, :, iRp_G...))

            for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
                Ω = value(meshes(Πpp, Val(1))[iΩ])
                ν = value(meshes(Πpp, Val(2))[iν])

                Πpp_R[iΩ, iν, iR_Π..., iRmR_Π...] += G_R(Ω - ν) * G_Rp(ν) * weight
                Πph_R[iΩ, iν, iR_Π..., iRpR_Π...] += G_R(Ω + ν) * G_Rp(ν) * weight
            end
        end
    end

    bfft!(Πpp_R, (3, 4, 5, 6))
    bfft!(Πph_R, (3, 4, 5, 6))

    return nothing
end
