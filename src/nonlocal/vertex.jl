abstract type AbstractNonlocalVertex{Q} <: AbstractVertex{Q} end

struct NL_Vertex{Q, VT} <: AbstractNonlocalVertex{Q}
    F0 :: VT
    γp :: NL_Channel{Q}
    γt :: NL_Channel{Q}
    γa :: NL_Channel{Q}

    function NL_Vertex(
        F0 :: VT,
        γp :: NL_Channel{Q},
        γt :: NL_Channel{Q},
        γa :: NL_Channel{Q},
        )  :: NL_Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    function NL_Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
        ) where {VT}

        Q = eltype(F0)

        γ = NL_Channel(T, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: NL_Vertex{Q}
    end
end

channel_type(::Type{NL_Vertex}) = NL_Channel

function Base.show(io::IO, Γ::NL_Vertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(bare_vertex(Γ)), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

# getter methods
function get_P_mesh(
    F :: AbstractNonlocalVertex
    ) :: KMesh

    return get_P_mesh(F.γp)
end

function numP(
    F :: AbstractNonlocalVertex
    ) :: Int64

    return numP(F.γp)
end

# copy
function Base.:copy(
    F :: NL_Vertex{Q}
    ) :: NL_Vertex{Q} where {Q}

    return NL_Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

# evaluators for parallel spin component
@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)
    mk  = meshes(F.γp.K1, Val(2))

    if F0
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    if γp
        val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)...,
                    convert_momentum( P, k, kp, Ch, pCh)...)
    end

    if γt
        val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)...,
                    convert_momentum( P, k, kp, Ch, tCh)...)
    end

    if γa
        val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)...,
                    convert_momentum( P, k, kp, Ch, aCh)...)
    end

    return val
end



# Special cases where either ν or νp is an InfiniteMatsubaraFrequency
@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    # This function is called only if either ν or νp is an InfiniteMatsubaraFrequency.
    # Otherwise, the specific case of having all MatsubaraFrequency's is called.

    # If ν or νp is an InfiniteMatsubaraFrequency, reducible vertices is nonzero
    # only for the same channel evaluated.

    if Ch === pCh && γp
        val += F.γp(Ω, ν, νp, P, k, kp)
    end

    if Ch === tCh && γt
        val += F.γt(Ω, ν, νp, P, k, kp)
    end

    if Ch === aCh && γa
        val += F.γa(Ω, ν, νp, P, k, kp)
    end

    return val
end


# evaluators for crossed spin component
@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{pCh},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    if kp isa SWaveBrillouinPoint
        Pmkp = SWaveBrillouinPoint()
    else
        Pmkp = P - kp
    end

    return -F(Ω, ν, Ω - νp, P, k, Pmkp, pCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)
end

@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{tCh},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    return -F(Ω, νp, ν, P, kp, k, aCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)
end

@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{aCh},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    return -F(Ω, νp, ν, P, kp, k, tCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)
end

# evaluators for density spin component
@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Ch},
       :: Type{dSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    val += F(Ω, ν, νp, P, k, kp, Ch, pSp; F0, γp, γt, γa) * 2

    val += F(Ω, ν, νp, P, k, kp, Ch, xSp; F0, γp, γt, γa)

    return val
end


# S wave evaluation

@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    kp :: BrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    # k isa SWaveBrillouinPoint, sum over the k momentum.
    # We use the fact that NL_Vertex has only bosonic momentum dependence.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., P)
        else
            # Vertices in the different channel will be integrated over the bosonic momentum.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., SWaveBrillouinPoint())
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., P)
        else
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., SWaveBrillouinPoint())
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., P)
        else
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., SWaveBrillouinPoint())
        end
    end

    return val
end

@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: SWaveBrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    # kp isa SWaveBrillouinPoint, sum over the kp momentum.
    # We use the fact that NL_Vertex has only bosonic momentum dependence.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., P)
        else
            # Vertices in the different channel will be integrated over the bosonic momentum.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., SWaveBrillouinPoint())
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., P)
        else
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., SWaveBrillouinPoint())
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., P)
        else
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., SWaveBrillouinPoint())
        end
    end

    return val
end

@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    kp :: SWaveBrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    # k and kp are SWaveBrillouinPoint, sum over the k and kp momentum.
    # We use the fact that NL_Vertex has only bosonic momentum dependence.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., P)
        else
            # Vertices in the different channel will be integrated over the bosonic momentum.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., SWaveBrillouinPoint())
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., P)
        else
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., SWaveBrillouinPoint())
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., P)
        else
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., SWaveBrillouinPoint())
        end
    end

    return val
end
