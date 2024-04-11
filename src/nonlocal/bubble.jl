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
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    G   :: NL_MF_G{Q},
    ) :: Nothing where {Q}

    # G(R) = ∑_k G(k) exp(-ikR) / N_k
    # Πpp(R, R') = G(R) * G(R' + R)
    # Πph(R, R') = G(R) * G(R' - R)
    # Π(P, k) = ∑_{R, r} Π(R, R') exp(iPR) exp(ikR')

    set!(Πpp, 0)
    set!(Πph, 0)

    LG = bz(meshes(G, Val(2))).L
    L  = bz(meshes(Πpp, Val(3))).L

    n1 = length(meshes(Πpp, Val(1)))
    n2 = length(meshes(Πpp, Val(2)))

    G_real_space = fft(reshape(G.data, :, LG, LG), (2, 3)) / LG^2

    Πpp_PR = zeros(eltype(Πpp.data), n1, n2, L^2, L, L)
    Πph_PR = zeros(eltype(Πph.data), n1, n2, L^2, L, L)

    # Scan R for Π in [-L/2, L/2]²
    Rs_Π_1d = (-div(L, 2)) : div(L, 2)
    Rs = collect(Iterators.product(Rs_Π_1d, Rs_Π_1d))

    for (Rp1, Rp2) in Rs
        Rp_vec = (Rp1, Rp2)

        # Index of Rp in Π(R, Rp)
        iRp_Π = mod.(Rp_vec, (L, L)) .+ 1

        Πpp_RR = zeros(eltype(Πpp.data), n1, n2, L, L)
        Πph_RR = zeros(eltype(Πph.data), n1, n2, L, L)

        for (R1, R2) in Rs
            R_vec = (R1, R2)

            weight = 1.0
            if mod(L, 2) == 0
                abs(Rp1) == div(L, 2) && (weight /= 2)
                abs(Rp2) == div(L, 2) && (weight /= 2)
                abs(R1) == div(L, 2) && (weight /= 2)
                abs(R2) == div(L, 2) && (weight /= 2)
            end

            # Index of R in Π(R, Rp)
            iR_Π  = mod.(R_vec,  (L, L)) .+ 1

            # Index of R, Rp + R, and Rp - R in G(R)
            iR_G = mod.(R_vec, (LG, LG)) .+ 1
            iRpR_G = mod.(Rp_vec .+ R_vec, (LG, LG)) .+ 1
            iRmR_G = mod.(Rp_vec .- R_vec, (LG, LG)) .+ 1

            G_R   = MeshFunction((meshes(G, Val(1)),), view(G_real_space, :, iR_G...))
            G_RpR = MeshFunction((meshes(G, Val(1)),), view(G_real_space, :, iRpR_G...))
            G_RmR = MeshFunction((meshes(G, Val(1)),), view(G_real_space, :, iRmR_G...))

            for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
                Ω = value(meshes(Πpp, Val(1))[iΩ])
                ν = value(meshes(Πpp, Val(2))[iν])

                Πpp_RR[iΩ, iν, iR_Π...] += G_R(Ω - ν) * G_RpR(ν) * weight
                Πph_RR[iΩ, iν, iR_Π...] += G_R(Ω + ν) * G_RmR(ν) * weight
            end
        end

        Πpp_PR[:, :, :, iRp_Π...] .+= reshape(bfft(Πpp_RR, (3, 4)), n1, n2, L^2)
        Πph_PR[:, :, :, iRp_Π...] .+= reshape(bfft(Πph_RR, (3, 4)), n1, n2, L^2)
    end

    Πpp.data .= reshape(bfft(Πpp_PR, (4, 5)), n1, n2, L^2, L^2)
    Πph.data .= reshape(bfft(Πph_PR, (4, 5)), n1, n2, L^2, L^2)

    return nothing
end
