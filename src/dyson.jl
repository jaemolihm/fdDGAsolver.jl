function Dyson!(S :: Union{ParquetSolver, NL_ParquetSolver}) :: Nothing
    Dyson!(S.G, S.Σ, S.Gbare)
end

function Dyson!(
    G     :: MF_G,
    Σ     :: MF_G,
    Gbare :: MF_G,
    ) :: Nothing

    for iν in eachindex(meshes(G, 1))
        ν = value(meshes(G, 1)[iν])
        # G and Σ actually store im * G and im * Σ, so -Σ in the Dyson equation becomes +Σ.
        G[ν] = 1 / (1 / Gbare[ν] + Σ(ν))
    end

    return nothing
end

function Dyson!(
    G     :: NL_MF_G,
    Σ     :: NL_MF_G,
    Gbare :: NL_MF_G,
    ) :: Nothing

    for ik in eachindex(meshes(G, 2))
        k = value(meshes(G, 2)[ik])
        for iν in eachindex(meshes(G, 1))
            ν = value(meshes(G, 1)[iν])

            # G and Σ is im * G and im * Σ, so -Σ in the Dyson equation becomes +Σ.
            G[ν, k] = 1 / (1 / Gbare[ν, k] + Σ(ν, k))
        end
    end

    return nothing
end
