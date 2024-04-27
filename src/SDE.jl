# Define AbstractSolver interface

function SDE!(S :: AbstractSolver; strategy = :scPA, include_U² = true, include_Hartree = true)
    if strategy == :scPA
        # Σ = SDE(ΔΓ + Γ₀, Π, G)
        set!(S.Σ, 0)
        SDE!(S.Σ, S.G, S.Πpp, S.Πph, S.Lpp, S.Lph, S.F, S.SGΣ, S.SGpp[2], S.SGph[2]; include_U², include_Hartree, S.mode)

    elseif strategy == :fdPA
        # Σ = SDE(ΔΓ + Γ₀, Π, G)
        set!(S.Σ, 0)
        SDE!(S.Σ, S.G, S.Πpp, S.Πph, S.Lpp, S.Lph, S.F, S.SGΣ, S.SGpp[2], S.SGph[2]; include_U², include_Hartree, S.mode)
        #   - SDE(Γ₀, Π₀, G₀)
        mult_add!(S.Σ, SDE!(copy(S.Σ)*0, S.G0, S.Π0pp, S.Π0ph, S.L0pp, S.L0ph, S.F0, S.SGΣ, S.SG0pp2, S.SG0ph2; include_U², include_Hartree, S.mode), -1)
        # #   + Σ₀
        add!(S.Σ, S.Σ0)

        # # Using K12
        # SDE_using_K12!(S)
        # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G - S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
        # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
    else
        throw(ArgumentError("Calculation strategy $strategy unknown"))
    end
end

function SDE!(Σ, G, Πpp, Πph, Lpp, Lph, F, SGΣ, SGpp, SGph; include_U² = true, include_Hartree = true, mode)
    # Apply SDE for F and all of its reference vertices F.F0, F.F0.F0, ... recursively.

    # Current vertex
    add!(Σ, SDE_compute!(copy(Σ), G, Πpp, Πph, Lpp, Lph, F, SGΣ, SGpp, SGph; mode, include_U², include_Hartree))

    if F isa AbstractVertex
        # RefVertex of the current vertex
        # Set include_U² and include_Hartree to false because it is already computed.
        SDE!(Σ, G, Πpp, Πph, Lpp, Lph, F.F0, SGΣ, SGpp, SGph; include_U² = false, include_Hartree = false, mode)
    end

    return Σ
end


# function SDE!(Σ, S :: AbstractSolver; include_U² = true, include_Hartree = true)
#     SDE_compute!(Σ, S.G, S.Πpp, S.Πph, S.Lpp, S.Lph, S.F, S.SGΣ, S.SGpp[2], S.SGph[2]; S.mode, include_U², include_Hartree)
#     return Σ
# end

# function SDE(S :: AbstractSolver; include_U² = true, include_Hartree = true)
#     SDE!(copy(S.Σ), S; include_U², include_Hartree)
# end

function SDE_U2(S :: AbstractSolver)
    SDE_U2_using_G(S)
end

function SDE_U2_using_Π(S :: AbstractSolver)
    SDE_U2_using_Π(S.Σ, S.G, S.Πpp, S.Πph, S.SGΣ, bare_vertex(S.F); S.mode)
end

function SDE_U2_using_G(S :: AbstractSolver)
    SDE_U2_using_G(S.Σ, S.G, S.SGΣ, bare_vertex(S.F); S.mode)
end

function SDE_using_K12!(S :: AbstractSolver; include_Hartree = true)
    SDE_using_K12!(S.Σ, S.G, S.F, S.SGΣ; S.mode, include_Hartree)
end


# Implementations for ParquetSolver (local vertex)

function SDE_channel_L_pp!(
    Lpp   :: MF_K2{Q},
    Πpp   :: MF_Π{Q},
    F     :: Vertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i])

            val += bare_vertex(F) * Πslice[i] * F.γp(Ω, Ω - ω, ν)
        end

        return temperature(F) * val
    end

    # compute Lpp
    SGpp2(Lpp, InitFunction{2, Q}(diagram); mode = mode)

    return nothing
end

function SDE_channel_L_ph!(
    Lph   :: MF_K2{Q},
    Πph   :: MF_Π{Q},
    F     :: Vertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i])

            val += bare_vertex(F) * Πslice[i] * (F.γt(Ω, ν, ω) + F.γa(Ω, ν, ω))
        end

        return temperature(F) * val
    end

    # compute Lph
    SGph2(Lph, InitFunction{2, Q}(diagram); mode)

    return nothing
end

# U * Π * (F - U) for F <: RefVertex

function SDE_channel_L_pp!(
    Lpp   :: MF_K2{Q},
    Πpp   :: MF_Π{Q},
    F     :: RefVertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i])

            val += bare_vertex(F) * Πslice[i] * (F(Ω, Ω - ω, ν, pCh, pSp) - bare_vertex(F, pCh, pSp))
        end

        return temperature(F) * val
    end

    # compute Lpp
    SGpp2(Lpp, InitFunction{2, Q}(diagram); mode = mode)

    return nothing
end


function SDE_channel_L_ph!(
    Lph   :: MF_K2{Q},
    Πph   :: MF_Π{Q},
    F     :: RefVertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i])

            val += bare_vertex(F) * Πslice[i] * (F(Ω, ν, ω, aCh, pSp) + F(Ω, ν, ω, tCh, pSp) - bare_vertex(F, aCh, pSp) - bare_vertex(F, tCh, pSp))
        end

        return temperature(F) * val
    end

    # compute Lph
    SGph2(Lph, InitFunction{2, Q}(diagram); mode)

    return nothing
end

function SDE_compute!(
    Σ     :: MF_G{Q},
    G     :: MF_G{Q},
    Πpp   :: MF_Π{Q},
    Πph   :: MF_Π{Q},
    Lpp   :: MF_K2{Q},
    Lph   :: MF_K2{Q},
    F     :: Union{Vertex{Q}, RefVertex{Q}},
    SGΣ   :: SymmetryGroup,
    SGpp2 :: SymmetryGroup,
    SGph2 :: SymmetryGroup,
    ;
    mode  :: Symbol,
    include_U² = true,
    include_Hartree = true,
    )     :: MF_G{Q} where {Q}
    # γa, γp, γt contribution to the self-energy in the asymptotic decomposition

    SDE_channel_L_pp!(Lpp, Πpp, F, SGpp2; mode)
    SDE_channel_L_ph!(Lph, Πph, F, SGph2; mode)

    # model the diagram
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        if is_inbounds(ν, meshes(Lpp, Val(2)))
            Lppslice = view(Lpp, :, ν)
            Lphslice = view(Lph, :, ν)

            for i in eachindex(Lppslice)
                Ω = value(meshes(Lpp, Val(1))[i])

                val += G(Ω - ν) * Lppslice[i]
                val += G(Ω + ν) * Lphslice[i]
            end
        end

        return temperature(F) * val
    end

    # compute Σ
    SGΣ(Σ, InitFunction{1, Q}(diagram); mode)

    if F isa RefVertex
        # If F is a RefVertex, a, p, and t channel contributions are all equivalent and
        # counted three times. We divide by 3 to get the correct result.
        Σ.data .*= 1/3
    end

    if include_U²
        Σ_U² = SDE_U2_using_G(Σ, G, SGΣ, bare_vertex(F); mode)
        add!(Σ, Σ_U²)
    end

    if include_Hartree
        n = compute_occupation(G)
        # We store im * Σ in S.Σ, so we multiply im.
        Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
    end

    return Σ
end


function SDE_U2_using_Π(
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

    T = temperature(meshes(Σ, Val(1)))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    Πppsum = MeshFunction((meshes(Πpp, Val(1)),), dropdims(sum(Πpp.data, dims=2), dims=2))
    Πphsum = MeshFunction((meshes(Πph, Val(1)),), dropdims(sum(Πph.data, dims=2), dims=2))

    # model the diagram
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        for i in eachindex(meshes(Πppsum, Val(1)))

            Ω = value(meshes(Πppsum, Val(1))[i])
            val += G(Ω - ν) * Πppsum[Ω] + G(Ω + ν) * Πphsum[Ω]

        end

        return val * U^2 * T^2 / 2
    end

    # compute Σ
    SGΣ(Σ_U², InitFunction{1, Q}(diagram); mode)

    return Σ_U²
end


function SDE_U2_using_G(
    Σ   :: MF_G{Q},
    G   :: MF_G{Q},
    SGΣ :: SymmetryGroup,
    U   :: Number,
    ;
    mode  :: Symbol,
    )   :: MF_G{Q} where {Q}
    # 2nd-order perturbative contribution to the self-energy

    T = temperature(meshes(Σ, Val(1)))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    # model the diagram
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        for i2 in eachindex(meshes(G, Val(1))), i1 in eachindex(meshes(G, Val(1)))
            ω1 = value(meshes(G, Val(1))[i1])
            ω2 = value(meshes(G, Val(1))[i2])

            if is_inbounds(ω1 - ω2 + ν, meshes(G, Val(1)))
                val += G[ω1] * G[ω2] * G[ω1 - ω2 + ν]
            end
        end

        return val * U^2 * T^2
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

        for iω in eachindex(meshes(G, Val(1)))
            ω = value(meshes(G, Val(1))[iω])
            # SDE using only K1 + K2 in p channel
            val += G[ω] * (F.γp.K1(ν + ω) + F.γp.K2(ν + ω, ν))
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
    for ν in meshes(Σ, Val(1))
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
