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

function green_with_tail(G :: MeshFunction{1, Q}, ν :: MatsubaraFrequency{Fermion}) :: Q where {Q}
    # Evaluate Green function G using the tail 1 / (im * ν) for extrapolation
    # Since we factor out (-im) factor in the Green function, the tail is 1 / ν.
    return is_inbounds(ν, meshes(G, Val(1))) ? G[ν] : 1 / value(ν)
    # return is_inbounds(ν, meshes(G, Val(1))) ? G[ν] : zero(Q)
end

function green_with_tail(G :: MeshFunction{2, Q}, ν :: MatsubaraFrequency{Fermion}, k :: BrillouinPoint) :: Q where {Q}
    return is_inbounds(ν, meshes(G, Val(1))) ? G(ν, k) : 1 / value(ν)
end

function _evaluate_G(G :: MeshFunction{2, Q}, ν, k_itp) :: Q where {Q}
    k_indices = indices(k_itp)
    k_weights = weights(k_itp)

    val = Q(0)
    for (i, w) in zip(k_indices, k_weights)
        val += w * G[ν, i]
    end
    val
end

function green_with_tail(G :: MeshFunction{2, Q}, ν :: MatsubaraFrequency{Fermion}, k :: SVector{2}) :: Q where {Q}
    if is_inbounds(ν, meshes(G, Val(1)))
        k_itp = MatsubaraFunctions.InterpolationParam(k, meshes(G, Val(2)))
        return _evaluate_G(G, ν, k_itp)
    else
        return 1 / value(ν)
    end

end


function bubbles_real_space!(
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    G1  :: NL_MF_G{Q},
    G2  :: NL_MF_G{Q} = G1
    ;
    use_G_tail :: Bool = true
    ) where {Q}

    # G(R) = ∑_k G(k) exp(-ikR) / N_k
    # Πpp(R) = G1(R) * G2( R)
    # Πph(R) = G1(R) * G2(-R)
    # Π(P) = ∑_{R} Π(R) exp(iPR)

    set!(Πpp, 0)
    set!(Πph, 0)

    LG = bz(meshes(G1, Val(2))).L
    L  = bz(meshes(Πpp, Val(3))).L

    n1 = length(meshes(Πpp, Val(1)))
    n2 = length(meshes(Πpp, Val(2)))

    G1_real_space = fft(reshape(G1.data, :, LG, LG), (2, 3)) / LG^2
    G2_real_space = fft(reshape(G2.data, :, LG, LG), (2, 3)) / LG^2

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

        G1_pR = MeshFunction((meshes(G1, Val(1)),), view(G1_real_space, :, ipR_G...))
        G2_pR = MeshFunction((meshes(G2, Val(1)),), view(G2_real_space, :, ipR_G...))
        G2_mR = MeshFunction((meshes(G2, Val(1)),), view(G2_real_space, :, imR_G...))

        for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
            Ω = value(meshes(Πpp, Val(1))[iΩ])
            ν = value(meshes(Πpp, Val(2))[iν])

            if R_vec == (0, 0) && use_G_tail
                Πpp_R[iΩ, iν, iR_Π...] += green_with_tail(G1_pR, Ω - ν) * green_with_tail(G2_pR, ν) * weight
                Πph_R[iΩ, iν, iR_Π...] += green_with_tail(G1_pR, Ω + ν) * green_with_tail(G2_mR, ν) * weight
            else
                Πpp_R[iΩ, iν, iR_Π...] += G1_pR(Ω - ν) * G2_pR(ν) * weight
                Πph_R[iΩ, iν, iR_Π...] += G1_pR(Ω + ν) * G2_mR(ν) * weight
            end
        end
    end

    bfft!(Πpp_R, (3, 4))
    bfft!(Πph_R, (3, 4))

    return nothing
end
