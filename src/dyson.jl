function Dyson!(S :: AbstractSolver)
    Dyson!(S.G, S.Σ, S.Gbare)
end

function Dyson!(
    G     :: MF_G,
    Σ     :: MF_G,
    Gbare :: MF_G,
    ) :: Nothing

    for ν in meshes(G, Val(1))
        # G and Σ actually store im * G and im * Σ, so -Σ in the Dyson equation becomes +Σ.
        G[ν] = 1 / (1 / Gbare[ν] + Σ[ν])
    end

    return nothing
end

function Dyson!(
    G     :: NL_MF_G,
    Σ     :: NL_MF_G,
    Gbare :: NL_MF_G,
    ) :: Nothing

    for k in meshes(G, Val(2)), ν in meshes(G, Val(1))
        # G and Σ is im * G and im * Σ, so -Σ in the Dyson equation becomes +Σ.
        G[ν, k] = 1 / (1 / Gbare[ν, k] + Σ[ν, k])
    end

    return nothing
end
