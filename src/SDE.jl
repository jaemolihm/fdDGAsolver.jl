function Dyson!(
    S :: ParquetSolver
    ) :: Nothing

    Δ = π / 5
    D = 10.0
    e = 0.
    for ν in value.(meshes(S.G, 1))
        # S.G and S.Σ is im * G and im * Σ, so -Σ in the Dyson equation becomes +Σ.
        S.G[ν] = 1 / (1 / S.Gbare[ν] + S.Σ(ν))
    end

    return nothing
end

function bubbles!(
    S :: ParquetSolver
    ) :: Nothing

    for Ω in value.(meshes(S.Πpp, 1)), ν in value.(meshes(S.Πpp, 2))
        S.Πpp[Ω, ν] = S.G(ν) * S.G(Ω - ν)
        S.Πph[Ω, ν] = S.G(Ω + ν) * S.G(ν)
    end

    return nothing
end

function SDE_channel_L_pp(
    S :: ParquetSolver{Q},
    ) :: MF_K2{Q} where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(S.Πpp, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(S.Π0pp, 2)[i])

            val += S.F.F0.U * Πslice[i] * S.F.γp(Ω, Ω - ω, ν)
        end

        return temperature(S) * val
    end

    # compute Lpp
    Lpp = copy(S.F.γp.K2)

    S.SGpp[2](Lpp, InitFunction{2, Q}(diagram); mode = S.mode)

    return Lpp
end

function SDE_channel_L_ph(
    S :: ParquetSolver{Q},
    ) :: MF_K2{Q} where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(S.Πph, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(S.Π0pp, 2)[i])

            val += S.F.F0.U * Πslice[i] * (S.F.γt(Ω, ν, ω) + S.F.γa(Ω, ν, ω))
        end

        return temperature(S) * val
    end

    # compute Lph
    Lph = copy(S.F.γt.K2)

    S.SGph[2](Lph, InitFunction{2, Q}(diagram); mode = S.mode)

    return Lph
end


function SDE!(
    S :: ParquetSolver{Q}
    ;
    include_U² = true,
    include_Hartree = true,
    ) :: Nothing where {Q}
    # γa, γp, γt contribution to the self-energy in the asymptotic decomposition

    Lpp = SDE_channel_L_pp(S)
    Lph = SDE_channel_L_ph(S)

    # model the diagram
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        if is_inbounds(ν, meshes(Lpp, 2))
            Lppslice = view(Lpp, :, ν)
            Lphslice = view(Lph, :, ν)

            for i in eachindex(Lppslice)
                Ω = value(meshes(Lpp, 1)[i])

                val += S.G(Ω - ν) * Lppslice[i]
                val += S.G(Ω + ν) * Lphslice[i]
            end
        end

        return temperature(S) * val
    end

    # compute Σ
    S.SGΣ(S.Σ, InitFunction{1, Q}(diagram); mode = S.mode)

    if include_U²
        Σ_U² = SDE_U2(S)
        add!(S.Σ, Σ_U²)
    end

    if include_Hartree
        n = compute_occupation(S.G)
        # We store im * Σ in S.Σ, so we multiply im.
        S.Σ.data .+= Q((n - 1/2) * S.F.F0.U * im)
    end

    return nothing
end

function SDE_U2(
    S :: ParquetSolver{Q},
    ) :: MF_G{Q} where {Q}
    # 2nd-order perturbative contribution to the self-energy

    Σ_U² = copy(S.Σ)
    set!(Σ_U², 0)

    # model the diagram
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        for Ω in value.(meshes(S.Πph, 1))
            Πppsum = sum(view(S.Πpp, Ω, :)) * temperature(S)
            Πphsum = sum(view(S.Πph, Ω, :)) * temperature(S)

            val += S.G(Ω - ν) * Πppsum + S.G(Ω + ν) * Πphsum
        end

        return temperature(S) * val * (S.F.F0.U)^2 / 2
    end

    # compute Σ
    S.SGΣ(Σ_U², InitFunction{1, Q}(diagram); mode = S.mode)

    return Σ_U²
end

function SDE_using_K12!(
    S :: ParquetSolver{Q}
    ;
    include_Hartree = true,
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        for ω in value.(meshes(S.G, 1))
            # SDE using only K1 + K2 in p channel
            val += S.G[ω] * (box_eval(S.F.γp.K1, ν + ω) + box_eval(S.F.γp.K2, ν + ω, ν))
            if S.F0 isa Vertex
                val += S.G[ω] * (box_eval(S.F0.γp.K1, ν + ω) + box_eval(S.F0.γp.K2, ν + ω, ν))
            end
        end

        return temperature(S) * val
    end

    # compute Σ
    S.SGΣ(S.Σ, InitFunction{1, Q}(diagram); mode = S.mode)

    if include_Hartree
        n = compute_occupation(S.G)
        # We store im * Σ in S.Σ, so we multiply im.
        S.Σ.data .+= Q((n - 1/2) * S.F.F0.U * im)
    end

    # sanity check
    self_energy_sanity_check(S.Σ)

    return nothing
end

function self_energy_sanity_check(Σ)
    passed = true
    # sanity check
    for ν in value.(meshes(Σ, 1))
        if value(ν) > 0 && real(Σ[ν]) < 0
            passed = false
            @warn "Σ violates causality at n = $(index(ν))"
        elseif value(ν) < 0 && real(Σ[ν]) > 0
            passed = false
            @warn "Σ violates causality at n = $(index(ν))"
        end
    end
    passed
end

function compute_occupation(
    G :: MF_G
) :: Float64
    return 0.5 + imag(sum(G.data)) * temperature(meshes(G, 1))
end

export
    compute_occupation
