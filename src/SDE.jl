# Define AbstractSolver interface

function SDE_channel_L_pp(S :: AbstractSolver)
    SDE_channel_L_pp(S.Πpp, S.F, S.SGpp[2]; S.mode)
end

function SDE_channel_L_ph(S :: AbstractSolver)
    SDE_channel_L_ph(S.Πph, S.F, S.SGph[2]; S.mode)
end

function SDE!(S :: AbstractSolver; include_U² = true, include_Hartree = true)
    SDE!(S.Σ, S.G, S.Πpp, S.Πph, S.F, S.SGΣ, S.SGpp[2], S.SGph[2]; S.mode, include_U², include_Hartree)
end

function SDE_U2(S :: AbstractSolver)
    SDE_U2(S.Σ, S.G, S.Πpp, S.Πph, S.SGΣ, bare_vertex(S.F); S.mode)
end

function SDE_using_K12!(S :: AbstractSolver; include_Hartree = true)
    SDE_using_K12!(S.Σ, S.G, S.F, S.SGΣ; S.mode, include_Hartree)
end


# Implementations for ParquetSolver (local vertex)

function SDE_channel_L_pp(
    Πpp   :: MF_Π{Q},
    F     :: Vertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: MF_K2{Q} where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, 2)[i])

            val += bare_vertex(F) * Πslice[i] * F.γp(Ω, Ω - ω, ν)
        end

        return temperature(F) * val
    end

    # compute Lpp
    Lpp = copy(F.γp.K2)

    SGpp2(Lpp, InitFunction{2, Q}(diagram); mode = mode)

    return Lpp
end

function SDE_channel_L_ph(
    Πph   :: MF_Π{Q},
    F     :: Vertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: MF_K2{Q} where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, 2)[i])

            val += bare_vertex(F) * Πslice[i] * (F.γt(Ω, ν, ω) + F.γa(Ω, ν, ω))
        end

        return temperature(F) * val
    end

    # compute Lph
    Lph = copy(F.γt.K2)

    SGph2(Lph, InitFunction{2, Q}(diagram); mode)

    return Lph
end


function SDE!(
    Σ     :: MF_G{Q},
    G     :: MF_G{Q},
    Πpp   :: MF_Π{Q},
    Πph   :: MF_Π{Q},
    F     :: Vertex{Q},
    SGΣ   :: SymmetryGroup,
    SGpp2 :: SymmetryGroup,
    SGph2 :: SymmetryGroup,
    ;
    mode  :: Symbol,
    include_U² = true,
    include_Hartree = true,
    )     :: MF_G{Q} where {Q}
    # γa, γp, γt contribution to the self-energy in the asymptotic decomposition

    Lpp = SDE_channel_L_pp(Πpp, F, SGpp2; mode)
    Lph = SDE_channel_L_ph(Πph, F, SGph2; mode)

    # model the diagram
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        if is_inbounds(ν, meshes(Lpp, 2))
            Lppslice = view(Lpp, :, ν)
            Lphslice = view(Lph, :, ν)

            for i in eachindex(Lppslice)
                Ω = value(meshes(Lpp, 1)[i])

                val += G(Ω - ν) * Lppslice[i]
                val += G(Ω + ν) * Lphslice[i]
            end
        end

        return temperature(F) * val
    end

    # compute Σ
    SGΣ(Σ, InitFunction{1, Q}(diagram); mode)

    if include_U²
        Σ_U² = SDE_U2(Σ, G, Πpp, Πph, SGΣ, bare_vertex(F); mode)
        add!(Σ, Σ_U²)
    end

    if include_Hartree
        n = compute_occupation(G)
        # We store im * Σ in S.Σ, so we multiply im.
        Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
    end

    return Σ
end


function SDE_U2(
    Σ   :: MF_G{Q},
    G   :: MF_G{Q},
    Πpp :: MF_Π{Q},
    Πph :: MF_Π{Q},
    SGΣ :: SymmetryGroup,
    U   :: Number,
    ;
    mode  :: Symbol,
    )   :: MF_G{Q} where {Q}
    # 2nd-order perturbative contribution to the self-energy

    T = temperature(meshes(Σ, 1))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    Πppsum = MeshFunction((meshes(Πpp, 1),), dropdims(sum(Πpp.data, dims=2), dims=2))
    Πphsum = MeshFunction((meshes(Πph, 1),), dropdims(sum(Πph.data, dims=2), dims=2))

    # model the diagram
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        for i in eachindex(meshes(Πppsum, 1))

            Ω = value(meshes(Πppsum, 1)[i])
            val += G(Ω - ν) * Πppsum[Ω] + G(Ω + ν) * Πphsum[Ω]

        end

        return val * U^2 * T^2 / 2
    end

    # compute Σ
    SGΣ(Σ_U², InitFunction{1, Q}(diagram); mode)

    return Σ_U²
end


function SDE_using_K12!(
    Σ :: MF_G{Q},
    G :: MF_G{Q},
    F :: Vertex{Q},
    SGΣ :: SymmetryGroup,
    ;
    mode :: Symbol,
    include_Hartree = true,
    ) :: MF_G{Q} where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        for iω in eachindex(meshes(G, 1))
            ω = value(meshes(G, 1)[iω])
            # SDE using only K1 + K2 in p channel
            val += G[ω] * (box_eval(F.γp.K1, ν + ω) + box_eval(F.γp.K2, ν + ω, ν))
        end

        return temperature(F) * val
    end

    # compute Σ
    SGΣ(Σ, InitFunction{1, Q}(diagram); mode)

    if include_Hartree
        n = compute_occupation(G)
        # We store im * Σ in S.Σ, so we multiply im.
        Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
    end

    return Σ
end

function self_energy_sanity_check(Σ)
    passed = true
    # sanity check
    for ν in meshes(Σ, 1)
        if plain_value(ν) > 0 && real(Σ[ν]) < 0
            passed = false
            @warn "Σ violates causality at n = $(index(ν))"
        elseif plain_value(ν) < 0 && real(Σ[ν]) > 0
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
