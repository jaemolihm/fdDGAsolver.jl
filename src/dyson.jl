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

function compute_occupation(
    G :: MF_G
) :: Float64
    return 0.5 + imag(sum(G.data)) * temperature(meshes(G, Val(1)))
end

function compute_occupation(G :: NL_MF_G)
    return 0.5 + imag(sum(G.data)) * temperature(meshes(G, Val(1))) / length(meshes(G, Val(2)))
end


# Compute chemical potential of the Hubbard model that gives the given the occupation
function compute_hubbard_chemical_potential(occ_target, Σ, hubbard_params)
    G_tmp = copy(Σ)

    function occupation(μ)
        Gbare = hubbard_bare_Green(meshes(Σ)...; μ, hubbard_params...)
        Dyson!(G_tmp, Σ, Gbare)
        compute_occupation(G_tmp)
    end

    μ = find_zero(μ -> occupation(μ) - occ_target, (-4 * abs(hubbard_params.t1), 4 * abs(hubbard_params.t1)))

    return μ
end
