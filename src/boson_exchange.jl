using fdDGAsolver: ChannelTag, AbstractVertex, Channel, bare_vertex, convert_frequency, convert_momentum, InfiniteMatsubaraFrequency, AbstractSolver, SpinTag, k0, MF_K2, MF_Π

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


# Evaluation for the given asymptotic class

@inline function (γ :: fdDGAsolver.Channel{Q})(
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

    else
        throw(ArgumentError("Invalid class tag $Cl"))
    end

end

@inline function (F :: RefVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: BrillouinPoint,
       :: BrillouinPoint,
       :: BrillouinPoint,
       :: Type{pCh},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K3Cl
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
       :: BrillouinPoint,
       :: BrillouinPoint,
       :: Type{tCh},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K3Cl
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
       :: BrillouinPoint,
       :: BrillouinPoint,
       :: Type{aCh},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K3Cl
        return -F.Ft_x(Ω, νp, ν)
    else
        return zero(Q)
    end
end



# -------------------------------------------------------------------------- #

struct MBEVertex{Q, NL, VT} <: AbstractVertex{Q}
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

        return new{Q, 0, VT}(F0, γp, γt, γa)
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
        return new{Q, 0, VT}(F0, γ, copy(γ), copy(γ))
    end
end

fdDGAsolver.channel_type(::Type{MBEVertex}) = Channel

@inline function (F :: Union{Channel{Q}, RefVertex{Q}, Vertex{Q}})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
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
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
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

function (F::MBEVertex{Q, 0})(
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

    return F(Ω, ν, νp, fdDGAsolver.k0, fdDGAsolver.k0, fdDGAsolver.k0, Ch, Sp; F0, γp, γt, γa)
end


function (F::MBEVertex{Q, 0})(
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
        val += - (K1 + K2 + K2p + K2 * K2p / (-U + K1) + K3) / 2

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

    if F0 == false
        val -= F.F0(Ω, ν, νp, P, k, kp, Ch, Sp; γp, γt, γa)
    end

    return val
end


@inline function (F :: MBEVertex{Q, 0})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
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
        return -F(Ω, νp, ν, P, kp, k, aCh, pSp; F0, γp, γt = γa, γa = γt)

    elseif Ch === aCh
        return -F(Ω, νp, ν, P, kp, k, tCh, pSp; F0, γp, γt = γa, γa = γt)

    else
        throw(ArgumentError("Invalid channel tag $Ch"))
    end
end


# evaluators for density spin component
@inline function (F :: MBEVertex{Q, 0})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
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
function fdDGAsolver.build_K3_cache!(
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

    # Subtract SBE contribution from K3
    U = bare_vertex(F_mbe)
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
