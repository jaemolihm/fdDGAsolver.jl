function bubbles!(S :: AbstractSolver)
    bubbles!(S.Πpp, S.Πph, S.G)
end

function bubbles!(
    Πpp :: MF_Π{Q},
    Πph :: MF_Π{Q},
    G   :: MF_G{Q},
    ) :: Nothing where {Q}

    for iΩ in eachindex(meshes(Πpp, 1)), iν in eachindex(meshes(Πpp, 2))
        Ω = value(meshes(Πpp, 1)[iΩ])
        ν = value(meshes(Πpp, 2)[iν])
        Πpp[iΩ, iν] = G(ν) * G(Ω - ν)
        Πph[iΩ, iν] = G(Ω + ν) * G(ν)
    end

    return nothing
end

function bubbles!(
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    G   :: NL_MF_G{Q},
    ) :: Nothing where {Q}

    set!(Πpp, 0)
    set!(Πph, 0)

    for iP in eachindex(meshes(Πpp, 3)), ik in eachindex(meshes(Πpp, 4))
        P = euclidean(meshes(Πpp, 3)[iP], meshes(Πpp, 3))
        k = euclidean(meshes(Πpp, 4)[ik], meshes(Πpp, 4))

        ind_k   = MatsubaraFunctions.mesh_index(    k, meshes(G, 2))
        ind_Pmk = MatsubaraFunctions.mesh_index(P - k, meshes(G, 2))
        ind_Ppk = MatsubaraFunctions.mesh_index(P + k, meshes(G, 2))

        for iΩ in eachindex(meshes(Πpp, 1)), iν in eachindex(meshes(Πpp, 2))
            Ω = value(meshes(Πpp, 1)[iΩ])
            ν = value(meshes(Πpp, 2)[iν])

            if is_inbounds(ν, meshes(G, 1))
                if is_inbounds(Ω - ν, meshes(G, 1))
                    Πpp[iΩ, iν, iP, ik] = G[ν, ind_k] * G[Ω - ν, ind_Pmk]
                end

                if is_inbounds(Ω + ν, meshes(G, 1))
                    Πph[iΩ, iν, iP, ik] = G[ν, ind_k] * G[Ω + ν, ind_Ppk]
                end
            end
        end
    end

    return nothing
end
