
struct ChannelViewX2X{Q, T1, T2, T3} <: AbstractReducibleVertex{Q}
    K1  :: T1
    K2  :: T2
    K2p :: T2
    K3  :: T3

    function ChannelViewX2X(
        K1  :: T1,
        K2  :: T2,
        K2p :: T2,
        K3  :: T3
        ) where {T1 <: MeshFunction{1, Q}, T2 <: MeshFunction{2, Q}, T3 <: MeshFunction{3, Q}} where {Q}

        return new{Q, T1, T2, T3}(K1, K2, K2p, K3)
    end
end


struct VertexViewX2X{Q, VT, CT} <: AbstractVertex{Q}
    F0  :: VT
    γpp :: CT
    γtp :: CT
    γap :: CT
    γpx :: CT
    γtx :: CT
    γax :: CT

    function VertexViewX2X(
        F0  :: VT,
        γpp :: CT,
        γtp :: CT,
        γap :: CT,
        γpx :: CT,
        γtx :: CT,
        γax :: CT,
        ) where {VT, CT <: ChannelViewX2X{Q}} where {Q}

        return new{Q, VT, CT}(F0, γpp, γtp, γap, γpx, γtx, γax)
    end
end

function MatsubaraFunctions.temperature(F :: VertexViewX2X) :: Float64
    return MatsubaraFunctions.temperature(F.γpp)
end

numK1(F :: VertexViewX2X) :: Int64 = numK1(F.γpp)
numK2(F :: VertexViewX2X) :: NTuple{2, Int64} = numK2(F.γpp)
numK3(F :: VertexViewX2X) :: NTuple{2, Int64} = numK3(F.γpp)


@inline function fixed_momentum_view(
    γ :: NL_Channel,
    P :: BrillouinPoint,
    k :: BrillouinPoint,
    q :: BrillouinPoint,
    )
    iP = MatsubaraFunctions.mesh_index_bc(P, get_P_mesh(γ))

    K1  = MeshFunction(γ.K1.meshes[1:1], view(γ.K1, :, iP))
    K2  = MeshFunction(γ.K2.meshes[1:2], view(γ.K2, :, :, iP))
    K2p = MeshFunction(γ.K2.meshes[1:2], view(γ.K2, :, :, iP))
    K3  = MeshFunction(γ.K3.meshes[1:3], view(γ.K3, :, :, :, iP))
    ChannelViewX2X(K1, K2, K2p, K3)
end


@inline function fixed_momentum_view(
    γ :: NL2_Channel,
    P :: BrillouinPoint,
    k :: BrillouinPoint,
    q :: BrillouinPoint,
    )
    iP = MatsubaraFunctions.mesh_index_bc(P, get_P_mesh(γ))
    ik = MatsubaraFunctions.mesh_index_bc(k, get_P_mesh(γ))
    iq = MatsubaraFunctions.mesh_index_bc(q, get_P_mesh(γ))

    K1  = MeshFunction(γ.K1.meshes[1:1], view(γ.K1, :, iP))
    K2  = MeshFunction(γ.K2.meshes[1:2], view(γ.K2, :, :, iP, ik))
    K2p = MeshFunction(γ.K2.meshes[1:2], view(γ.K2, :, :, iP, iq))
    K3  = MeshFunction(γ.K3.meshes[1:3], view(γ.K3, :, :, :, iP))
    ChannelViewX2X(K1, K2, K2p, K3)
end

@inline function fixed_momentum_view(
    γ :: Union{Vertex, RefVertex},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch}
    ) where {Ch <: ChannelTag}
    γ
end


@inline function fixed_momentum_view(
    F  :: Union{NL_Vertex, NL2_Vertex},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch}
    ) where {Ch <: ChannelTag}

    F0 = fixed_momentum_view(F.F0, P, k, kp, Ch)

    ks_p = convert_momentum(P, k, kp, Ch, pCh)
    ks_t = convert_momentum(P, k, kp, Ch, tCh)
    ks_a = convert_momentum(P, k, kp, Ch, aCh)

    γpp = fixed_momentum_view(F.γp, ks_p...)
    γtp = fixed_momentum_view(F.γt, ks_t...)
    γap = fixed_momentum_view(F.γa, ks_a...)

    # γpx, γtx, and γax stores vertices with crossing applied to the momentum argument.
    if Ch === pCh
        γpx = fixed_momentum_view(F.γp, convert_momentum(P, k, P - kp, pCh, pCh)...)
        γtx = fixed_momentum_view(F.γt, convert_momentum(P, k, P - kp, pCh, tCh)...)
        γax = fixed_momentum_view(F.γa, convert_momentum(P, k, P - kp, pCh, aCh)...)
    elseif Ch === tCh
        γpx = fixed_momentum_view(F.γp, convert_momentum(P, kp, k, aCh, pCh)...)
        γtx = fixed_momentum_view(F.γt, convert_momentum(P, kp, k, aCh, tCh)...)
        γax = fixed_momentum_view(F.γa, convert_momentum(P, kp, k, aCh, aCh)...)
    elseif Ch === aCh
        γpx = fixed_momentum_view(F.γp, convert_momentum(P, kp, k, tCh, pCh)...)
        γtx = fixed_momentum_view(F.γt, convert_momentum(P, kp, k, tCh, tCh)...)
        γax = fixed_momentum_view(F.γa, convert_momentum(P, kp, k, tCh, aCh)...)
    end

    VertexViewX2X(F0, γpp, γtp, γap, γpx, γtx, γax)
end


@inline function (γ :: ChannelViewX2X{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency}
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    if is_inbounds(Ω, meshes(γ.K1, Val(1)))
        K1 && (val += γ.K1[Ω])

        if is_inbounds(Ω, meshes(γ.K2, Val(1)))
            ν1_inbounds = is_inbounds(ν,  meshes(γ.K2, Val(2)))
            ν2_inbounds = is_inbounds(νp, meshes(γ.K2, Val(2)))

            if ν1_inbounds && ν2_inbounds
                K2 && (val += γ.K2[Ω, ν] + γ.K2p[Ω, νp])
                K3 && (val += γ.K3(Ω, ν, νp))

            elseif ν1_inbounds
                K2 && (val += γ.K2[Ω, ν])

            elseif ν2_inbounds
                K2 && (val += γ.K2p[Ω, νp])
            end
        end
    end

    return val
end


@inline function (F :: VertexViewX2X{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
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
        val += F.F0(Ω, ν, νp, Ch, pSp)
    end

    if γp
        val += F.γpp(convert_frequency(Ω, ν, νp, Ch, pCh)...)
    end

    if γt
        val += F.γtp(convert_frequency(Ω, ν, νp, Ch, tCh)...)
    end

    if γa
        val += F.γap(convert_frequency(Ω, ν, νp, Ch, aCh)...)
    end

    return val
end

@inline function (F :: VertexViewX2X{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{Ch},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    # Use crossing symmetry to evaluate crossed spin component.
    # γpx, γtx, and γax stores vertices with crossing applied to the momentum argument.

    if F0
        val += F.F0(Ω, ν, νp, Ch, xSp)
    end

    if Ch === pCh
        γp && (val -= F.γpx(convert_frequency(Ω, ν, Ω - νp, pCh, pCh)...))
        γt && (val -= F.γax(convert_frequency(Ω, ν, Ω - νp, pCh, aCh)...))
        γa && (val -= F.γtx(convert_frequency(Ω, ν, Ω - νp, pCh, tCh)...))

    elseif Ch === tCh
        γp && (val -= F.γpx(convert_frequency(Ω, νp, ν, aCh, pCh)...))
        γt && (val -= F.γax(convert_frequency(Ω, νp, ν, aCh, aCh)...))
        γa && (val -= F.γtx(convert_frequency(Ω, νp, ν, aCh, tCh)...))

    elseif Ch === aCh
        γp && (val -= F.γpx(convert_frequency(Ω, νp, ν, tCh, pCh)...))
        γt && (val -= F.γax(convert_frequency(Ω, νp, ν, tCh, aCh)...))
        γa && (val -= F.γtx(convert_frequency(Ω, νp, ν, tCh, tCh)...))

    end

    return val
end

# Special cases where either ν or νp is an InfiniteMatsubaraFrequency
@inline function (F :: VertexViewX2X{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
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
        val += F.F0(Ω, ν, νp, Ch, pSp)
    end

    # This function is called only if either ν or νp is an InfiniteMatsubaraFrequency.
    # Otherwise, the specific case of having all MatsubaraFrequency's is called.

    # If ν or νp is an InfiniteMatsubaraFrequency, reducible vertices is nonzero
    # only for the same channel evaluated.

    if Ch === pCh && γp
        val += F.γpp(Ω, ν, νp)
    end

    if Ch === tCh && γt
        val += F.γtp(Ω, ν, νp)
    end

    if Ch === aCh && γa
        val += F.γap(Ω, ν, νp)
    end

    return val
end


# Special cases where either ν or νp is an InfiniteMatsubaraFrequency
@inline function (F :: VertexViewX2X{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, Ch, xSp)
    end

    # This function is called only if either ν or νp is an InfiniteMatsubaraFrequency.
    # Otherwise, the specific case of having all MatsubaraFrequency's is called.

    # If ν or νp is an InfiniteMatsubaraFrequency, reducible vertices is nonzero
    # only for the same channel evaluated.

    if Ch === pCh && γp
        val -= F.γpx(Ω, ν, Ω - νp)
    end

    if Ch === tCh && γt
        val -= F.γax(Ω, νp, ν)
    end

    if Ch === aCh && γa
        val -= F.γtx(Ω, νp, ν)
    end

    return val
end



# evaluators for density spin component
@inline function (F :: VertexViewX2X{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{dSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    val += F(Ω, ν, νp, Ch, pSp; F0, γp, γt, γa) * 2

    val += F(Ω, ν, νp, Ch, xSp; F0, γp, γt, γa)

    return val
end
