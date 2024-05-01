"""
    abstract type ClassTag end

Abstract type for asymptotic classes
"""
abstract type ClassTag end

"""
    struct K1Cl <: ClassTag end

K1 class
"""
struct K1Cl <: ClassTag end

"""
    struct K2Cl <: ClassTag end

K2 class
"""
struct K2Cl <: ClassTag end

"""
    struct K2pCl <: ClassTag end

K2' class
"""
struct K2pCl <: ClassTag end

"""
    struct K3Cl <: ClassTag end

K3 class
"""
struct K3Cl <: ClassTag end

"""
    struct ΛCl <: ClassTag end

Λ class. 2-particle irreducible vertex.
"""
struct ΛCl <: ClassTag end

_crossing(::Type{K1Cl}) = K1Cl
_crossing(::Type{K2Cl}) = K2pCl
_crossing(::Type{K2pCl}) = K2Cl
_crossing(::Type{K3Cl}) = K3Cl
_crossing(::Type{ΛCl}) = ΛCl


# Evaluation for the given asymptotic class

@inline function (γ :: Channel{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: BrillouinPoint,
       :: BrillouinPoint,
       :: BrillouinPoint,
       :: Type{Cl},
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K1Cl
        return γ.K1(Ω)

    elseif Cl === K2Cl
        return γ.K2(Ω, ν)

    elseif Cl === K2pCl
        return γ.K2(Ω, νp)

    elseif Cl === K3Cl
        return γ.K3(Ω, ν, νp)

    elseif Cl === ΛCl
        return zero(Q)

    else
        throw(ArgumentError("Invalid class tag $Cl"))
    end

end

@inline function (γ :: Channel{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Cl},
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K1Cl
        return γ.K1(Ω)

    elseif Cl === K2Cl
        return γ.K2(Ω, ν)

    elseif Cl === K2pCl
        return γ.K2(Ω, νp)

    elseif Cl === K3Cl
        return γ.K3(Ω, ν, νp)

    elseif Cl === ΛCl
        return zero(Q)

    else
        throw(ArgumentError("Invalid class tag $Cl"))
    end

end

@inline function (γ :: ChannelViewX2X{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Cl},
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K1Cl
        return γ.K1(Ω)

    elseif Cl === K2Cl
        return γ.K2(Ω, ν)

    elseif Cl === K2pCl
        return γ.K2p(Ω, νp)

    elseif Cl === K3Cl
        return γ.K3(Ω, ν, νp)

    elseif Cl === ΛCl
        return zero(Q)

    else
        throw(ArgumentError("Invalid class tag $Cl"))
    end

end


@inline function (γ :: NL2_Channel{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Cl},
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K1Cl
        return γ.K1(Ω, P)

    elseif Cl === K2Cl
        return γ.K2(Ω, ν, P, k)

    elseif Cl === K2pCl
        return γ.K2(Ω, νp, P, kp)

    elseif Cl === K3Cl
        return γ.K3(Ω, ν, νp, P)

    elseif Cl === ΛCl
        return zero(Q)

    else
        throw(ArgumentError("Invalid class tag $Cl"))
    end

end

@inline function (F :: RefVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: BrillouinPoint,
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{pCh},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === ΛCl
        return F.Fp_p(Ω, ν, νp)
    else
        return zero(Q)
    end
end

@inline function (F :: RefVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: BrillouinPoint,
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{tCh},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === ΛCl
        return F.Ft_p(Ω, ν, νp)
    else
        return zero(Q)
    end
end




@inline function (F :: RefVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: BrillouinPoint,
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{aCh},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === ΛCl
        return -F.Ft_x(Ω, νp, ν)
    else
        return zero(Q)
    end
end



# -------------------------------------------------------------------------- #

abstract type AbstractMBEVertex{Q} <: AbstractVertex{Q} end

struct MBEVertex{Q, VT} <: AbstractMBEVertex{Q}
    F0 :: VT
    γp :: Channel{Q}
    γt :: Channel{Q}
    γa :: Channel{Q}

    function MBEVertex(
        F0 :: VT,
        γp :: Channel{Q},
        γt :: Channel{Q},
        γa :: Channel{Q},
        ) where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    function MBEVertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        ) where {VT}

        Q = eltype(F0)

        γ = Channel(T, numK1, numK2, numK3, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ))
    end
end

channel_type(::Type{MBEVertex}) = Channel

@inline function (F :: Union{Channel{Q}, RefVertex{Q}, Vertex{Q}})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Ch},
       :: Type{Cl},
    )  :: Q where {Q, Ch <: ChannelTag, Cl <: ClassTag}

    # Recursively evaluate the Vertex `F` and its RefVertex `F.F0`
    # at the pSp spin component, channel `Ch` and asymptotic class `Cl`

    val = F.F0(Ω, ν, νp, P, k, kp, Ch, Cl)

    if Ch === pCh
        val += F.γp(Ω, ν, νp, P, k, kp, Cl)
    end

    if Ch === tCh
        val += F.γt(Ω, ν, νp, P, k, kp, Cl)
    end

    if Ch === aCh
        val += F.γa(Ω, ν, νp, P, k, kp, Cl)
    end

    return val
end

@inline function (F :: AbstractVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Ch},
       :: Type{Cl},
    )  :: Q where {Q, Ch <: ChannelTag, Cl <: ClassTag}

    # Recursively evaluate the Vertex `F` and its RefVertex `F.F0`
    # at the pSp spin component, channel `Ch` and asymptotic class `Cl`

    val = F.F0(Ω, ν, νp, P, k, kp, Ch, Cl)

    if Ch === pCh
        val += F.γp(Ω, ν, νp, P, k, kp, Cl)
    end

    if Ch === tCh
        val += F.γt(Ω, ν, νp, P, k, kp, Cl)
    end

    if Ch === aCh
        val += F.γa(Ω, ν, νp, P, k, kp, Cl)
    end

    return val
end

function (F::MBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{Sp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    return F(Ω, ν, νp, k0, k0, k0, Ch, Sp; F0, γp, γt, γa)
end


function (F::AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch},
       :: Type{Sp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: pSp}

    val = zero(Q)

    U = bare_vertex(F, Sp)

    val += U

    if γp
        ωs = convert_frequency(Ω, ν, νp, Ch, pCh)
        ks = convert_momentum( P, k, kp, Ch, pCh)
        K1  = F(ωs..., ks..., pCh, K1Cl)
        K2  = F(ωs..., ks..., pCh, K2Cl)
        K2p = F(ωs..., ks..., pCh, K2pCl)
        K3  = F(ωs..., ks..., pCh, K3Cl)
        val += K1 + K2 + K2p + K2 * K2p / (U + K1) + K3
    end

    if γt
        # tCh, pSp =  1/2 * (tCh, D) - 1/2 * (tCh, M)
        # (tCh, M) = - (aCh, pSp)
        # (tCh, D) = 2 * (tCh, pSp) - (aCh, pSp)
        ωs = convert_frequency(Ω, ν, νp, Ch, tCh)
        ks = convert_momentum( P, k, kp, Ch, tCh)

        # Magnetic channel
        K1  = - F(ωs..., ks..., aCh, K1Cl)
        K2  = - F(ωs..., ks..., aCh, K2Cl)
        K2p = - F(ωs..., ks..., aCh, K2pCl)
        K3  = - F(ωs..., ks..., aCh, K3Cl)
        val -= (K1 + K2 + K2p + K2 * K2p / (-U + K1) + K3) / 2

        # Density channel
        K1  = 2 * F(ωs..., ks..., tCh, K1Cl)  + K1
        K2  = 2 * F(ωs..., ks..., tCh, K2Cl)  + K2
        K2p = 2 * F(ωs..., ks..., tCh, K2pCl) + K2p
        K3  = 2 * F(ωs..., ks..., tCh, K3Cl)  + K3
        val += (K1 + K2 + K2p + K2 * K2p / (U + K1) + K3) / 2
    end

    if γa
        ωs = convert_frequency(Ω, ν, νp, Ch, aCh)
        ks = convert_momentum( P, k, kp, Ch, aCh)
        K1  = F(ωs..., ks..., aCh, K1Cl)
        K2  = F(ωs..., ks..., aCh, K2Cl)
        K2p = F(ωs..., ks..., aCh, K2pCl)
        K3  = F(ωs..., ks..., aCh, K3Cl)

        val += K1 + K2 + K2p + K2 * K2p / (U + K1) + K3
    end

    # Add 2-particle irreducible contribution
    val += F.F0(Ω, ν, νp, P, k, kp, Ch, ΛCl)

    if F0 == false
        val -= F.F0(Ω, ν, νp, P, k, kp, Ch, Sp; γp, γt, γa)
    end

    return val
end


@inline function (F :: AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Ch},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    # Use crossing symmetry to map xSp to pSp.

    if Ch === pCh
        return -F(Ω, ν, Ω - νp, P, k, P - kp, pCh, pSp; F0, γp, γt = γa, γa = γt)

    elseif Ch === tCh
        return -F(Ω, ν, νp, P, k, kp, aCh, pSp; F0, γp, γt = γa, γa = γt)

    elseif Ch === aCh
        return -F(Ω, ν, νp, P, k, kp, tCh, pSp; F0, γp, γt = γa, γa = γt)

    else
        throw(ArgumentError("Invalid channel tag $Ch"))
    end
end


# evaluators for density spin component
@inline function (F :: AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: Union{BrillouinPoint, SWaveBrillouinPoint},
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

    return ( 2 * F(Ω, ν, νp, P, k, kp, Ch, pSp; F0, γp, γt, γa)
               + F(Ω, ν, νp, P, k, kp, Ch, xSp; F0, γp, γt, γa) )
end



function (F::AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: SWaveBrillouinPoint,
       :: Type{Ch},
       :: Type{pSp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    mK = get_P_mesh(F)

    for q in mK
        val += F(Ω, ν, νp, P, k, value(q), Ch, pSp; F0, γp, γt, γa)
    end

    return val / length(mK)
end

function (F::AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch},
       :: Type{pSp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    mK = get_P_mesh(F)

    for q in mK
        val += F(Ω, ν, νp, P, value(q), kp, Ch, pSp; F0, γp, γt, γa)
    end

    return val / length(mK)
end

function (F::AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    kp :: SWaveBrillouinPoint,
       :: Type{Ch},
       :: Type{pSp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    mK = get_P_mesh(F)

    for qp in mK, q in mK
        val += F(Ω, ν, νp, P, value(q), value(qp), Ch, pSp; F0, γp, γt, γa)
    end

    return val / length(mK)^2
end



function Base.:copy(
    F :: MBEVertex{Q}
    ) :: MBEVertex{Q} where {Q}

    return MBEVertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end



# -------------------------------------------------------------------------- #


function get_reducible_vertex(
    F :: AbstractVertex,
      :: Type{Ch},
    ) where {Ch <: ChannelTag}

    if Ch === aCh
        return F.γa
    elseif Ch === pCh
        return F.γp
    elseif Ch === tCh
        return F.γt
    else
        throw(ArgumentError("Invalid channel tag $Ch"))
    end
end



# There are multiple types of vertices to precompute.
# (1) `cache_F0*`: Channel-U-irreducible vertexof the reference system (`S.F0`)
#                  Cache ``Tᵣ = Γ - ∇ᵣ = Iᵣ - U + Mᵣ``.
# (2) `cache_Γ*` : Irreducible finite-difference vertex (`S.F` with `F0=false`, `γ*=false`)
# (3) `cache_F*` : Channel-U-irreducible vertexof the target system (`S.F`)
function build_K3_cache!(
    S :: ParquetSolver{Q, <: MBEVertex}
    ) where {Q}

    U = bare_vertex(S.F)

    set!(S.cache_Γpx, 0)
    set!(S.cache_F0p, 0)
    set!(S.cache_F0a, 0)
    set!(S.cache_F0t, 0)
    set!(S.cache_Γpp, 0)
    set!(S.cache_Γa,  0)
    set!(S.cache_Γt,  0)
    set!(S.cache_Fp,  0)
    set!(S.cache_Fa,  0)
    set!(S.cache_Ft,  0)

    # Vertices multiplied by bubbles to the left (by ω)
    # Γpx : Target, irreducible vertex in the p channel, xSp component.

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ω, νp = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        S.cache_Γpx[i] = S.F( Ω, ω, νp, pCh, xSp; F0=false, γp=false)

        # (Iᵣ - U) + Mᵣ
        S.cache_F0p[i] = S.F0(Ω, ω, νp, pCh, xSp; γp = false) + U + S.F0(Ω, ω, Ω - νp, k0, k0, k0, pCh, K3Cl) * -1
        S.cache_F0a[i] = S.F0(Ω, ω, νp, aCh, pSp; γa = false) - U + S.F0(Ω, ω, νp, k0, k0, k0, aCh, K3Cl)
        S.cache_F0t[i] = S.F0(Ω, ω, νp, tCh, pSp; γt = false) - U + S.F0(Ω, ω, νp, k0, k0, k0, tCh, K3Cl)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # Vertices multiplied by bubbles from the right (by ω)

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, ω = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # r-irreducible vertex in each channel r = p, a, t
        S.cache_Γpp[i] = S.F(Ω, ν, ω, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, tCh, pSp; F0=false, γt=false)

        # (Iᵣ - U) + Mᵣ
        S.cache_Fp[i]  = S.F(Ω, ν, ω, pCh, pSp; γp = false) - U + S.F(Ω, ν, ω, k0, k0, k0, pCh, K3Cl)
        S.cache_Fa[i]  = S.F(Ω, ν, ω, aCh, pSp; γa = false) - U + S.F(Ω, ν, ω, k0, k0, k0, aCh, K3Cl)
        S.cache_Ft[i]  = S.F(Ω, ν, ω, tCh, pSp; γt = false) - U + S.F(Ω, ν, ω, k0, k0, k0, tCh, K3Cl)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_Γt[i] = S.cache_Γt[i] * 2 - S.cache_Γa[i]
        S.cache_Ft[i] = S.cache_Ft[i] * 2 - S.cache_Fa[i]
    end

    mpi_allreduce!(S.cache_Γpp)
    mpi_allreduce!(S.cache_Γa)
    mpi_allreduce!(S.cache_Γt)
    mpi_allreduce!(S.cache_Fp)
    mpi_allreduce!(S.cache_Fa)
    mpi_allreduce!(S.cache_Ft)

    return nothing
end


function asymptotic_to_mbe(F :: Vertex)
    F_mbe = MBEVertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))

    U = bare_vertex(F_mbe)

    if numK3(F.γa) >= numK3(F.F0)
        # Subtract SBE contribution from K3
        for Ch in (aCh, pCh, tCh)
            γ = get_reducible_vertex(F_mbe, Ch)

            for i in eachindex(γ.K3.data)
                Ω, ν, ω = value.(to_meshes(γ.K3, i))
                K1 = γ.K1(Ω)
                K2 = γ.K2(Ω, ν)
                K2p = γ.K2(Ω, ω)
                ∇ = F_mbe(Ω, ν, ω, Ch, pSp; γp = (Ch === pCh), γt = (Ch === tCh), γa = (Ch === aCh)) - γ.K3(Ω, ν, ω)
                γ.K3.data[i] -= ∇ - (U + K1 + K2 + K2p)
            end
        end

    else
        # Subtract SBE contribution from RefVertex F0
        # Since F0 is the total vertex, not channel reducible, we need to subtract SBE
        # contributions in all channels.
        for (Λ, Ch, Sp) in [(F_mbe.F0.Fp_p, pCh, pSp),
                            (F_mbe.F0.Fp_x, pCh, xSp),
                            (F_mbe.F0.Ft_p, tCh, pSp),
                            (F_mbe.F0.Ft_x, tCh, xSp),]
            for i in eachindex(Λ.data)
                Ω, ν, ω = value.(to_meshes(Λ, i))

                ∇ = F_mbe(Ω, ν, ω, Ch, Sp; F0 = false) - F(Ω, ν, ω, Ch, Sp; F0 = false)
                Λ[i] -= ∇
            end
        end
    end

    return F_mbe
end


function mbe_to_asymptotic(F_mbe :: MBEVertex)
    F = Vertex(copy(F_mbe.F0), copy(F_mbe.γp), copy(F_mbe.γt), copy(F_mbe.γa))

    # Subtract SBE contribution from K3
    U = bare_vertex(F)
    for Ch in (aCh, pCh, tCh)
        γ = get_reducible_vertex(F, Ch)

        for i in eachindex(γ.K3.data)
            Ω, ν, ω = value.(to_meshes(γ.K3, i))
            K1 = γ.K1(Ω)
            K2 = γ.K2(Ω, ν)
            K2p = γ.K2(Ω, ω)
            K123 = F_mbe(Ω, ν, ω, Ch, pSp; γp = (Ch === pCh), γt = (Ch === tCh), γa = (Ch === aCh))
            γ.K3.data[i] = K123 - (U + K1 + K2 + K2p)
        end
    end

    return F
end



# -------------------------------------------------------------------------- #

struct NL2_MBEVertex{Q, VT} <: AbstractMBEVertex{Q}
    F0 :: VT
    γp :: NL2_Channel{Q}
    γt :: NL2_Channel{Q}
    γa :: NL2_Channel{Q}

    function NL2_MBEVertex(
        F0 :: VT,
        γp :: NL2_Channel{Q},
        γt :: NL2_Channel{Q},
        γa :: NL2_Channel{Q},
        ) where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    function NL2_MBEVertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
        ) where {VT}

        Q = eltype(F0)

        γ = NL2_Channel(T, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ))
    end
end

channel_type(::Type{NL2_MBEVertex}) = NL2_Channel

function Base.:copy(
    F :: NL2_MBEVertex{Q}
    ) :: NL2_MBEVertex{Q} where {Q}

    return NL2_MBEVertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

function get_P_mesh(
    F :: NL2_MBEVertex
    ) :: KMesh

    return get_P_mesh(F.γp)
end

function numP(
    F :: NL2_MBEVertex
    ) :: Int64

    return length(get_P_mesh(F))
end


function asymptotic_to_mbe(F :: NL2_Vertex)
    F_mbe = NL2_MBEVertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))

    # Subtract SBE contribution from K3
    U = bare_vertex(F_mbe)
    for Ch in (aCh, pCh, tCh)
        γ = get_reducible_vertex(F_mbe, Ch)

        for i in eachindex(γ.K3.data)
            Ω, ν, ω, P = value.(to_meshes(γ.K3, i))
            K1 = γ.K1(Ω, P)
            K2 = γ.K2(Ω, ν, P, kSW)
            K2p = γ.K2(Ω, ω, P, kSW)
            ∇ = F_mbe(Ω, ν, ω, P, kSW, kSW, Ch, pSp; γp = (Ch === pCh), γt = (Ch === tCh), γa = (Ch === aCh)) - γ.K3(Ω, ν, ω, P)
            γ.K3.data[i] -= ∇ - (U + K1 + K2 + K2p)
        end
    end

    return F_mbe
end


function mbe_to_asymptotic(F_mbe :: NL2_MBEVertex)
    F = NL2_Vertex(copy(F_mbe.F0), copy(F_mbe.γp), copy(F_mbe.γt), copy(F_mbe.γa))

    # Subtract SBE contribution from K3
    U = bare_vertex(F)
    for Ch in (aCh, pCh, tCh)
        γ = get_reducible_vertex(F, Ch)

        for i in eachindex(γ.K3.data)
            Ω, ν, ω, P = value.(to_meshes(γ.K3, i))
            K1 = γ.K1(Ω, P)
            K2 = γ.K2(Ω, ν, P, kSW)
            K2p = γ.K2(Ω, ω, P, kSW)
            K123 = F_mbe(Ω, ν, ω, P, kSW, kSW, Ch, pSp; γp = (Ch === pCh), γt = (Ch === tCh), γa = (Ch === aCh))
            γ.K3.data[i] = K123 - (U + K1 + K2 + K2p)
        end
    end

    return F
end



struct MBEVertexViewX2X{Q, VT, CT} <: AbstractVertex{Q}
    F0  :: VT
    γpp :: CT
    γtp :: CT
    γatp :: CT
    γaap :: CT
    γpx :: CT
    γtx :: CT
    γatx :: CT
    γaax :: CT

    function MBEVertexViewX2X(
        F0  :: VT,
        γpp :: CT,
        γtp :: CT,
        γatp :: CT,
        γaap :: CT,
        γpx :: CT,
        γtx :: CT,
        γatx :: CT,
        γaax :: CT,
        ) where {VT, CT <: ChannelViewX2X{Q}} where {Q}

        return new{Q, VT, CT}(F0, γpp, γtp, γatp, γaap, γpx, γtx, γatx, γaax)
    end
end

function MatsubaraFunctions.temperature(F :: MBEVertexViewX2X) :: Float64
    return MatsubaraFunctions.temperature(F.γpp)
end

numK1(F :: MBEVertexViewX2X) :: Int64 = numK1(F.γpp)
numK2(F :: MBEVertexViewX2X) :: NTuple{2, Int64} = numK2(F.γpp)
numK3(F :: MBEVertexViewX2X) :: NTuple{2, Int64} = numK3(F.γpp)


@inline function (F :: MBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{Ch_k},
       :: Type{Sp},
       :: Type{Cl},
    )  :: Q where {Q, Ch <: ChannelTag, Ch_k <: ChannelTag, Sp <: SpinTag, Cl <: ClassTag}

    # Recursively evaluate the Vertex `F` and its RefVertex `F.F0`
    # at the `Sp` spin component, channel `Ch` and asymptotic class `Cl`

    val = F.F0(Ω, ν, νp, Ch, Ch_k, Sp, Cl)

    Ch === pCh && Sp == pSp && (val += F.γp(Ω, ν, νp, Cl))
    Ch === pCh && Sp == xSp && (val -= F.γp(Ω, ν, νp, Cl))

    Ch === tCh && Sp == pSp && (val += F.γt(Ω, ν, νp, Cl))
    Ch === tCh && Sp == xSp && (val -= F.γt(Ω, ν, νp, Cl))

    Ch === aCh && Sp == pSp && (val += F.γa(Ω, ν, νp, Cl))
    Ch === aCh && Sp == xSp && (val -= F.γa(Ω, ν, νp, Cl))

    return val
end



@inline function (F :: RefVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{Ch_k},
       :: Type{Sp},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag, Ch_k <: ChannelTag, Sp <: SpinTag, Cl <: ClassTag}

    if Cl === ΛCl
        if Ch === pCh && Sp === pSp
            return F.Fp_p(Ω, ν, νp)

        elseif Ch === tCh && Sp === pSp
            return F.Ft_p(Ω, ν, νp)

        elseif Ch === aCh && Sp === pSp
            return -F.Ft_x(Ω, νp, ν)

        elseif Ch === pCh && Sp === xSp
            return -F.Fp_p(Ω, ν, Ω - νp)

        elseif Ch === tCh && Sp === xSp
            return F.Ft_x(Ω, ν, νp)

        elseif Ch === aCh && Sp === xSp
            return -F.Ft_p(Ω, νp, ν)

        end
    else
        return zero(Q)
    end
end



# There are multiple types of vertices to precompute.
# (1) `cache_F0*`: Channel-U-irreducible vertexof the reference system (`S.F0`)
#                  Cache ``Tᵣ = Γ - ∇ᵣ = Iᵣ - U + Mᵣ``.
# (2) `cache_Γ*` : Irreducible finite-difference vertex (`S.F` with `F0=false`, `γ*=false`)
# (3) `cache_F*` : Channel-U-irreducible vertexof the target system (`S.F`)
function build_K3_cache!(
    S :: NL2_ParquetSolver{Q, <: NL2_MBEVertex}
    ) where {Q}

    U = bare_vertex(S.F)

    set!(S.cache_Γpx, 0)
    set!(S.cache_F0p, 0)
    set!(S.cache_F0a, 0)
    set!(S.cache_F0t, 0)
    set!(S.cache_Γpp, 0)
    set!(S.cache_Γa,  0)
    set!(S.cache_Γt,  0)
    set!(S.cache_Fp,  0)
    set!(S.cache_Fa,  0)
    set!(S.cache_Ft,  0)

    # Vertices multiplied by bubbles to the left (by ω)
    # Γpx : Target, irreducible vertex in the p channel, xSp component.

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ω, νp, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        S.cache_Γpx[i] = S.F( Ω, ω, νp, P, kSW, kSW, pCh, xSp; F0=false, γp=false)

        # (Iᵣ - U) + Mᵣ
        S.cache_F0p[i] = S.F0(Ω, ω, νp, P, kSW, kSW, pCh, xSp; γp = false) + U + S.F0(Ω, ω, Ω - νp, P, kSW, kSW, pCh, K3Cl) * -1
        S.cache_F0a[i] = S.F0(Ω, ω, νp, P, kSW, kSW, aCh, pSp; γa = false) - U + S.F0(Ω, ω, νp, P, kSW, kSW, aCh, K3Cl)
        S.cache_F0t[i] = S.F0(Ω, ω, νp, P, kSW, kSW, tCh, pSp; γt = false) - U + S.F0(Ω, ω, νp, P, kSW, kSW, tCh, K3Cl)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # Vertices multiplied by bubbles from the right (by ω)

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, ω, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # r-irreducible vertex in each channel r = p, a, t
        S.cache_Γpp[i] = S.F(Ω, ν, ω, P, kSW, kSW, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, P, kSW, kSW, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, P, kSW, kSW, tCh, pSp; F0=false, γt=false)

        # (Iᵣ - U) + Mᵣ
        S.cache_Fp[i]  = S.F(Ω, ν, ω, P, kSW, kSW, pCh, pSp; γp = false) - U + S.F(Ω, ν, ω, P, kSW, kSW, pCh, K3Cl)
        S.cache_Fa[i]  = S.F(Ω, ν, ω, P, kSW, kSW, aCh, pSp; γa = false) - U + S.F(Ω, ν, ω, P, kSW, kSW, aCh, K3Cl)
        S.cache_Ft[i]  = S.F(Ω, ν, ω, P, kSW, kSW, tCh, pSp; γt = false) - U + S.F(Ω, ν, ω, P, kSW, kSW, tCh, K3Cl)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_Γt[i] = S.cache_Γt[i] * 2 - S.cache_Γa[i]
        S.cache_Ft[i] = S.cache_Ft[i] * 2 - S.cache_Fa[i]
    end

    mpi_allreduce!(S.cache_Γpp)
    mpi_allreduce!(S.cache_Γa)
    mpi_allreduce!(S.cache_Γt)
    mpi_allreduce!(S.cache_Fp)
    mpi_allreduce!(S.cache_Fa)
    mpi_allreduce!(S.cache_Ft)

    return nothing
end
