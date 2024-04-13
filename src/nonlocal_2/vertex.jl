struct NL2_Vertex{Q, VT} <: AbstractNonlocalVertex{Q}
    F0 :: VT
    γp :: NL2_Channel{Q}
    γt :: NL2_Channel{Q}
    γa :: NL2_Channel{Q}

    function NL2_Vertex(
        F0 :: VT,
        γp :: NL2_Channel{Q},
        γt :: NL2_Channel{Q},
        γa :: NL2_Channel{Q},
        )  :: NL2_Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    function NL2_Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
        ) where {VT}

        Q = eltype(F0)

        γ = NL2_Channel(T, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: NL2_Vertex{Q}
    end
end

channel_type(::Type{NL2_Vertex}) = NL2_Channel


function Base.show(io::IO, Γ::NL2_Vertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(bare_vertex(Γ.F0)), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

# copy
function Base.:copy(
    F :: NL2_Vertex{Q}
    ) :: NL2_Vertex{Q} where {Q}

    return NL2_Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end


# S wave evaluation

@inline function (F :: NL2_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    ω  :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    q  :: BrillouinPoint,
       :: Type{Ch},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, ω, P, k, q, Ch, pSp)
    end

    # k isa SWaveBrillouinPoint, sum over the k momentum.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(Ω, ν, ω, P, k, q)
        else
            # For vertices in the different channel,
            # K1 and K3 vertices have only bosonic momentum dependence and are fully integrated.
            ωs = convert_frequency(Ω, ν, ω, Ch, pCh)
            val += F.γp(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            # K2 vertex has two momentum dependence and only one linear combiniation
            # of the two is integrated. We manually compute the sum.
            for k_int in get_P_mesh(F)
                ks = convert_momentum(P, value(k_int), q, Ch, pCh)
                val += F.γp(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(Ω, ν, ω, P, k, q)
        else
            ωs = convert_frequency(Ω, ν, ω, Ch, tCh)
            val += F.γt(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            for k_int in get_P_mesh(F)
                ks = convert_momentum(P, value(k_int), q, Ch, tCh)
                val += F.γt(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(Ω, ν, ω, P, k, q)
        else
            ωs = convert_frequency(Ω, ν, ω, Ch, aCh)
            val += F.γa(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            for k_int in get_P_mesh(F)
                ks = convert_momentum(P, value(k_int), q, Ch, aCh)
                val += F.γa(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    return val
end

@inline function (F :: NL2_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    ω  :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    q  :: SWaveBrillouinPoint,
       :: Type{Ch},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, ω, P, k, q, Ch, pSp)
    end

    # q isa SWaveBrillouinPoint, sum over the q momentum.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(Ω, ν, ω, P, k, q)
        else
            # For vertices in the different channel,
            # K1 and K3 vertices have only bosonic momentum dependence and are fully integrated.
            ωs = convert_frequency(Ω, ν, ω, Ch, pCh)
            val += F.γp(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            # K2 vertex has two momentum dependence and only one linear combiniation
            # of the two is integrated. We manually compute the sum.
            for q_int in get_P_mesh(F)
                ks = convert_momentum(P, k, value(q_int), Ch, pCh)
                val += F.γp(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(Ω, ν, ω, P, k, q)
        else
            ωs = convert_frequency(Ω, ν, ω, Ch, tCh)
            val += F.γt(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            for q_int in get_P_mesh(F)
                ks = convert_momentum(P, k, value(q_int), Ch, tCh)
                val += F.γt(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(Ω, ν, ω, P, k, q)
        else
            ωs = convert_frequency(Ω, ν, ω, Ch, aCh)
            val += F.γa(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            for q_int in get_P_mesh(F)
                ks = convert_momentum(P, k, value(q_int), Ch, aCh)
                val += F.γa(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    return val
end

@inline function (F :: NL2_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    ω  :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    q  :: SWaveBrillouinPoint,
       :: Type{Ch},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, ω, P, k, q, Ch, pSp)
    end

    # k and q are SWaveBrillouinPoint, sum over the k and q momentum.
    # We use the fact that NL2_Vertex at most two momentum dependences.
    # Then, for different channels all momentum (bosonic for K1 and K3, bosonic and
    # fermionic for K2) are integrated.

    if γp
        if Ch === pCh
            # Vertices in the same channel: frequency and momentum arguments do not change
            val += F.γp(Ω, ν, ω, P, k, q)
        else
            # Vertices in the different channel will be integrated over the bosonic momentum.
            val += F.γp(convert_frequency(Ω, ν, ω, Ch, pCh)..., kSW, kSW, kSW)
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(Ω, ν, ω, P, k, q)
        else
            val += F.γt(convert_frequency(Ω, ν, ω, Ch, tCh)..., kSW, kSW, kSW)
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(Ω, ν, ω, P, k, q)
        else
            val += F.γa(convert_frequency(Ω, ν, ω, Ch, aCh)..., kSW, kSW, kSW)
        end
    end

    return val
end
