
"""
    bare_Green_Ω_flow(Λ, G0_imp, Σ_imp, G0_lat) => G0_Λ

Compute the bare Green function with the flow parameter `Λ`. `Λ = ∞` gives the impurity
Green function `G0_imp`, and `Λ = 0` gives the lattice Green function `G0_lat`.

DMFT self-consistency ``∫dk (G0_Λ(k, ν)⁻¹ - Σ_imp(ν))⁻¹ = G_imp(ν)`` is enforced.

Implements Eqs.(20, 23, 24) of PRResearch, 4, 013034 (2022).
"""
function bare_Green_Ω_flow(
    Λ      :: Float64,
    G0_imp :: MF_G{Q},
    Σ_imp  :: MF_G{Q},
    G0_lat :: NL_MF_G{Q}
    ) :: NL_MF_G{Q} where {Q}

    mG = meshes(G0_lat, Val(1))
    mK = meshes(G0_lat, Val(2))

    G_imp = copy(G0_imp)
    Dyson!(G_imp, Σ_imp, G0_imp)

    G0_Λ = copy(G0_lat)
    set!(G0_Λ, 0)

    for iν in eachindex(mG)
        ν = value(mG[iν])
        Θ = value(ν)^2 / (value(ν)^2 + Λ^2)

        function _dmft_self_consistency(Ξ)
            G0_Λ_ν = @. Θ * G0_lat[ν, :] + Ξ * G0_imp[ν]
            return sum(@. 1 / (1 / G0_Λ_ν + Σ_imp[ν])) / length(mK) - G_imp[ν]
        end

        res = nlsolve(x -> (y = _dmft_self_consistency(complex(x[1], x[2]));
            [real(y), imag(y)]), [0.0, 0.0], show_trace = false, ftol = 1e-12)
        Ξ = complex(res.zero...)

        G0_Λ.data[iν, :] .= @. Θ * G0_lat[ν, :] + Ξ * G0_imp[ν]
    end

    # Sanity check of DMFT self-consistency
    Σ_lat = copy(G0_Λ)
    G_Λ = copy(G0_Λ)
    set!(Σ_lat, 0)
    for iν in eachindex(mG)
        ν = value(mG[iν])
        Σ_lat.data[iν, :] .= Σ_imp(ν)
    end
    
    Dyson!(G_Λ, Σ_lat, G0_Λ)
    y1 = [G_imp[value(x)] for x in mG]
    y2 = sum(G_Λ.data, dims=2) ./ length(mK)
    @assert maximum(abs.(y1 .- y2)) < 1e-10
    # @info "L2  error = $(norm(y1 .- y2))"
    # @info "Max error = $(maximum(abs.(y1 .- y2)))"

    return G0_Λ
end
