function bubbles!(S :: AbstractSolver)
    bubbles!(S.Πpp, S.Πph, S.G)

    # Symmetrize
    # S.SGΠpp(S.Πpp)
    # S.SGΠph(S.Πph)
end

function bubbles!(
    Πpp :: MF_Π{Q},
    Πph :: MF_Π{Q},
    G   :: MF_G{Q},
    ) :: Nothing where {Q}

    for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
        Ω = value(meshes(Πpp, Val(1))[iΩ])
        ν = value(meshes(Πpp, Val(2))[iν])
        # Πpp[iΩ, iν] = G(ν) * G(Ω - ν)
        # Πph[iΩ, iν] = G(Ω + ν) * G(ν)
        G_ν   = is_inbounds(ν,     meshes(G, Val(1))) ? G[ν]     : 1 / value(ν)
        G_Ωmν = is_inbounds(Ω - ν, meshes(G, Val(1))) ? G[Ω - ν] : 1 / value(Ω - ν)
        G_Ωpν = is_inbounds(Ω + ν, meshes(G, Val(1))) ? G[Ω + ν] : 1 / value(Ω + ν)
        Πpp[iΩ, iν] = G_ν * G_Ωmν
        Πph[iΩ, iν] = G_ν * G_Ωpν
    end

    return nothing
end
