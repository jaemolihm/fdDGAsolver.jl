struct NL2_MBEChannel{Q}
    U :: Q
    η :: NL_MF_K1{Q}
    λ :: NL2_MF_K2{Q}
    M :: NL_MF_K3{Q}

    function NL2_MBEChannel(
        U :: Q,
        η :: NL_MF_K1{Q},
        λ :: NL2_MF_K2{Q},
        M :: NL_MF_K3{Q}
        ) where {Q}
        return new{Q}(U, η, λ, M)
    end

    function NL2_MBEChannel(
        T     :: Float64,
        U     :: Number,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
              :: Type{Q} = ComplexF64
        ) where {Q}

        mK1Ω = MatsubaraMesh(T, numK1, Boson)
        η    = MeshFunction(mK1Ω, meshK; data_t = Q)
        set!(η, U)

        @assert numK1 > numK2[1] "No. bosonic frequencies in K1 must be larger than no. bosonic frequencies in K2"
        @assert numK1 > numK2[2] "No. bosonic frequencies in K1 must be larger than no. fermionic frequencies in K2"
        mK2Ω = MatsubaraMesh(T, numK2[1], Boson)
        mK2ν = MatsubaraMesh(T, numK2[2], Fermion)
        λ    = MeshFunction(mK2Ω, mK2ν, meshK, meshK; data_t = Q)
        set!(λ, 1)

        @assert all(numK2 .>= numK3) "Number of frequencies in K2 must be larger than in K3"
        mK3Ω = MatsubaraMesh(T, numK3[1], Boson)
        mK3ν = MatsubaraMesh(T, numK3[2], Fermion)
        M    = MeshFunction(mK3Ω, mK3ν, mK3ν, meshK; data_t = Q)
        set!(M, 0)

        return new{Q}(Q(U), η, λ, M)
    end
end

MatsubaraFunctions.temperature(∇ :: NL2_MBEChannel) = MatsubaraFunctions.temperature(meshes(∇.η, Val(1)))
get_P_mesh(∇ :: NL2_MBEChannel) :: KMesh = meshes(∇.η, Val(2))


@inline function (∇ :: NL2_MBEChannel{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    q  :: BrillouinPoint,
    )  :: Q where {Q}

    iP = MatsubaraFunctions.mesh_index_bc(P, meshes(∇.η, Val(2)))
    ik = MatsubaraFunctions.mesh_index_bc(k, meshes(∇.η, Val(2)))
    iq = MatsubaraFunctions.mesh_index_bc(q, meshes(∇.η, Val(2)))

    η  = ∇.U
    λ  = one(Q)
    λb = one(Q)
    M = zero(Q)

    if is_inbounds(Ω, meshes(∇.η, Val(1)))
        η = ∇.η[Ω, iP]

        if is_inbounds(Ω, meshes(∇.λ, Val(1)))
            ν1_inbounds = is_inbounds(ν, meshes(∇.λ, Val(2)))
            ν2_inbounds = is_inbounds(ω, meshes(∇.λ, Val(2)))

            if ν1_inbounds
                λ = ∇.λ[Ω, ν, iP, ik]
            end

            if ν2_inbounds
                λb = ∇.λ[Ω, ω, iP, iq]
            end

            if ν1_inbounds && ν2_inbounds
                if ( is_inbounds(Ω, meshes(∇.M, Val(1))) &&
                     is_inbounds(ν, meshes(∇.M, Val(2))) &&
                     is_inbounds(ω, meshes(∇.M, Val(3))))
                    M = ∇.M[Ω, ν, ω, iP]
                end
            end
        end
    end

    return λ * η * λb + M - ∇.U
end



# ------------------------------------------------------------------------------------
struct NL2_MBEVertex{Q, VT} <: AbstractNonlocalVertex{Q}
    F0 :: VT
    ∇P :: NL2_MBEChannel{Q}
    ∇D :: NL2_MBEChannel{Q}
    ∇M :: NL2_MBEChannel{Q}

    function NL2_MBEVertex(
        F0 :: VT,
        ∇P :: NL2_MBEChannel{Q},
        ∇D :: NL2_MBEChannel{Q},
        ∇M :: NL2_MBEChannel{Q},
        )  :: NL2_MBEVertex{Q} where {Q, VT}

        return new{Q, VT}(F0, ∇P, ∇D, ∇M)
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
        U = bare_vertex(F0)

        ∇P = NL2_MBEChannel(T, 2U, numK1, numK2, numK3, meshK, Q)
        ∇D = NL2_MBEChannel(T,  U, numK1, numK2, numK3, meshK, Q)
        ∇M = NL2_MBEChannel(T, -U, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, ∇P, ∇D, ∇M) :: NL2_MBEVertex{Q}
    end
end

MatsubaraFunctions.temperature(F :: NL2_MBEVertex) = MatsubaraFunctions.temperature(F.∇P)

get_P_mesh(F :: NL2_MBEVertex) :: KMesh = get_P_mesh(F.∇P)

bare_vertex(∇ :: NL2_MBEVertex) = ∇.∇D.U

function numK1(F :: NL2_MBEVertex) :: Int64
    return length(meshes(F.∇P.η, Val(1)))
end

function numK2(F :: NL2_MBEVertex) :: NTuple{2, Int64}
    return (length(meshes(F.∇P.λ, Val(1))), length(meshes(F.∇P.λ, Val(2))))
end

function numK3(F :: NL2_MBEVertex) :: NTuple{2, Int64}
    return (length(meshes(F.∇P.M, Val(1))), length(meshes(F.∇P.M, Val(2))))
end


# evaluators for parallel spin component
@inline function (F :: NL2_MBEVertex{Q})(
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

    if F0
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    if γp
        val += F.∇P(convert_frequency(Ω, ν, νp, Ch, pCh)...,
                    convert_momentum( P, k, kp, Ch, pCh)...) / 2
    end

    if γt
        val += F.∇D(convert_frequency(Ω, ν, νp, Ch, tCh)...,
                    convert_momentum( P, k, kp, Ch, tCh)...) / 2
        val -= F.∇M(convert_frequency(Ω, ν, νp, Ch, tCh)...,
                    convert_momentum( P, k, kp, Ch, tCh)...) / 2
    end

    if γa
        val += F.∇M(convert_frequency(Ω, ν, νp, Ch, aCh)...,
                    convert_momentum( P, k, kp, Ch, aCh)...) * -1
    end

    return val
end

@inline function (F :: NL2_MBEVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
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

    if F0
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    if γp
        val += F.∇P(convert_frequency(Ω, ν, νp, Ch, pCh)...,
                    convert_momentum( P, k, kp, Ch, pCh)...) / 2
    end

    if γt
        val += F.∇D(convert_frequency(Ω, ν, νp, Ch, tCh)...,
                    convert_momentum( P, k, kp, Ch, tCh)...) / 2
        val -= F.∇M(convert_frequency(Ω, ν, νp, Ch, tCh)...,
                    convert_momentum( P, k, kp, Ch, tCh)...) / 2
    end

    if γa
        val += F.∇M(convert_frequency(Ω, ν, νp, Ch, aCh)...,
                    convert_momentum( P, k, kp, Ch, aCh)...) * -1
    end

    return val
end


@inline function (F :: NL2_MBEVertex{Q})(
    Ω :: MatsubaraFrequency,
    ν :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ω :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P :: BrillouinPoint,
    k :: SWaveBrillouinPoint,
    q :: SWaveBrillouinPoint,
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

    if γp
        ωs = convert_frequency(Ω, ν, ω, Ch, pCh)
        for q_int in get_P_mesh(F), k_int in get_P_mesh(F)
            ks = convert_momentum(P, value(k_int), value(q_int), Ch, pCh)
            val += F.∇P(ωs..., ks...) / numP(F)^2 / 2
        end
    end

    if γt
        ωs = convert_frequency(Ω, ν, ω, Ch, tCh)
        for q_int in get_P_mesh(F), k_int in get_P_mesh(F)
            ks = convert_momentum(P, value(k_int), value(q_int), Ch, tCh)
            val += F.∇D(ωs..., ks...) / numP(F)^2 / 2
            val -= F.∇M(ωs..., ks...) / numP(F)^2 / 2

        end
    end

    if γa
        ωs = convert_frequency(Ω, ν, ω, Ch, aCh)
        for q_int in get_P_mesh(F), k_int in get_P_mesh(F)
            ks = convert_momentum(P, value(k_int), value(q_int), Ch, aCh)
            val += F.∇M(ωs..., ks...) / numP(F)^2 * -1
        end
    end

    return val
end





function to_mbe(
    F :: NL2_Vertex{Q}
    ) :: NL2_MBEVertex{Q} where {Q}

    U = bare_vertex(F)

    # η = U + K1
    ηP = copy(F.γa.K1)
    ηD = copy(F.γa.K1)
    ηM = copy(F.γa.K1)

    for i in eachindex(ηP.data)
        Ω, P = value.(to_meshes(ηP, i))

        ηP[i] = F(Ω, νInf, νInf, P, k0, k0, pCh, pSp) * 2
        ηD[i] = F(Ω, νInf, νInf, P, k0, k0, tCh, dSp)
        ηM[i] = F(Ω, νInf, νInf, P, k0, k0, tCh, xSp)
    end

    λP = copy(F.γa.K2)
    λD = copy(F.γa.K2)
    λM = copy(F.γa.K2)

    for i in eachindex(λP.data)
        Ω, ν, P, k = value.(to_meshes(λP, i))

        λP[i] = F(Ω, ν, νInf, P, k, k0, pCh, pSp) * 2 / ηP[Ω, P]
        λD[i] = F(Ω, ν, νInf, P, k, k0, tCh, dSp) / ηD[Ω, P]
        λM[i] = F(Ω, ν, νInf, P, k, k0, tCh, xSp) / ηM[Ω, P]
    end

    MP = copy(F.γa.K3)
    MD = copy(F.γa.K3)
    MM = copy(F.γa.K3)

    for i in eachindex(MP.data)
        Ω, ν, ω, P = value.(to_meshes(MP, i))
        MP[i] = F.γp.K3(Ω, ν, ω, P) * 2
        MD[i] = F.γt.K3(Ω, ν, ω, P) * 2 - F.γa.K3(Ω, ν, ω, P)
        MM[i] = F.γa.K3(Ω, ν, ω, P) * -1
        MP[i] -= (λP(Ω, ν, P, kSW) - 1) * (λP(Ω, ω, P, kSW) - 1) * ηP(Ω, P)
        MD[i] -= (λD(Ω, ν, P, kSW) - 1) * (λD(Ω, ω, P, kSW) - 1) * ηD(Ω, P)
        MM[i] -= (λM(Ω, ν, P, kSW) - 1) * (λM(Ω, ω, P, kSW) - 1) * ηM(Ω, P)
    end

    ∇P = NL2_MBEChannel(2U, ηP, λP, MP)
    ∇D = NL2_MBEChannel( U, ηD, λD, MD)
    ∇M = NL2_MBEChannel(-U, ηM, λM, MM)

    NL2_MBEVertex(F.F0, ∇P, ∇D, ∇M)
end



function SDE_channel_L_pp!(
    Lpp   :: NL2_MF_K2{Q},
    Πpp   :: NL2_MF_Π{Q},
    ∇     :: NL2_MBEVertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Πslice = MeshFunction((meshes(Πpp, Val(2)), meshes(Πpp, Val(4))), view(Πpp, Ω, :, P, :))

        Πslice  = view(Πpp, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i.I[1]])
            q = value(meshes(Πpp, Val(4))[i.I[2]])

            val += ∇.∇P.U * Πslice[i] * ∇.∇P(Ω, ω, ν, P, k, q)
        end

        return val * temperature(∇.∇P) / length(meshes(Πpp, Val(4)))
    end

    # compute Lpp
    SGpp2(Lpp, InitFunction{4, Q}(diagram); mode = mode)

    Lpp.data .*= 0.25

    return nothing
end

function SDE_channel_L_ph!(
    Lph   :: NL2_MF_K2{Q},
    Πph   :: NL2_MF_Π{Q},
    ∇     :: NL2_MBEVertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    )     :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i.I[1]])
            q = value(meshes(Πph, Val(4))[i.I[2]])

            val += ∇.∇D.U * Πslice[i] * ∇.∇D(Ω, ν, ω, P, k, q) * 0.5
            val += ∇.∇M.U * Πslice[i] * ∇.∇M(Ω, ν, ω, P, k, q) * 1.5
        end

        return val * temperature(∇.∇D) / length(meshes(Πph, Val(4)))
    end

    # compute Lph
    SGph2(Lph, InitFunction{4, Q}(diagram); mode)

    return nothing
end






# -------------------------------------------------------------------------

struct MBEChannelViewX1X{Q, T1, T2, T3} <: AbstractReducibleVertex{Q}
    U  :: Q
    η  :: T1
    λ  :: T2
    λp :: T2
    M  :: T3

    function MBEChannelViewX1X(
        U  :: Q,
        η  :: T1,
        λ  :: T2,
        λp :: T2,
        M  :: T3
        ) where {T1 <: MeshFunction{1, Q}, T2 <: MeshFunction{2, Q}, T3 <: MeshFunction{3, Q}} where {Q}

        return new{Q, T1, T2, T3}(U, η, λ, λp, M)
    end
end


@inline function fixed_momentum_view(
    ∇ :: NL2_MBEChannel,
    P :: BrillouinPoint,
    k :: BrillouinPoint,
    q :: BrillouinPoint,
    )

    iP = MatsubaraFunctions.mesh_index_bc(P, get_P_mesh(∇))
    ik = MatsubaraFunctions.mesh_index_bc(k, get_P_mesh(∇))
    iq = MatsubaraFunctions.mesh_index_bc(q, get_P_mesh(∇))

    η  = MeshFunction(∇.η.meshes[1:1], view(∇.η, :, iP))
    λ  = MeshFunction(∇.λ.meshes[1:2], view(∇.λ, :, :, iP, ik))
    λp = MeshFunction(∇.λ.meshes[1:2], view(∇.λ, :, :, iP, iq))
    M  = MeshFunction(∇.M.meshes[1:3], view(∇.M, :, :, :, iP))
    MBEChannelViewX1X(∇.U, η, λ, λp, M)
end



@inline function (∇ :: MBEChannelViewX1X{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    )  :: Q where {Q}

    η  = ∇.U
    λ  = one(Q)
    λb = one(Q)
    M = zero(Q)

    if is_inbounds(Ω, meshes(∇.η, Val(1)))
        η = ∇.η[Ω]

        if is_inbounds(Ω, meshes(∇.λ, Val(1)))
            ν1_inbounds = is_inbounds(ν, meshes(∇.λ, Val(2)))
            ν2_inbounds = is_inbounds(ω, meshes(∇.λ, Val(2)))

            if ν1_inbounds
                λ = ∇.λ[Ω, ν]
            end

            if ν2_inbounds
                λb = ∇.λ[Ω, ω]
            end

            if ν1_inbounds && ν2_inbounds
                if ( is_inbounds(Ω, meshes(∇.M, Val(1))) &&
                     is_inbounds(ν, meshes(∇.M, Val(2))) &&
                     is_inbounds(ω, meshes(∇.M, Val(3))))
                    M = ∇.M[Ω, ν, ω]
                end
            end
        end
    end

    return λ * η * λb + M - ∇.U
end




struct MBEVertexViewX1X{Q, VT, CT} <: AbstractVertex{Q}
    F0  :: VT
    ∇Sp :: CT
    ∇Tp :: CT
    ∇Da :: CT
    ∇Dt :: CT
    ∇Ma :: CT
    ∇Mt :: CT

    function MBEVertexViewX1X(
        F0  :: VT,
        ∇Sp :: CT,
        ∇Tp :: CT,
        ∇Da :: CT,
        ∇Dt :: CT,
        ∇Ma :: CT,
        ∇Mt :: CT,
        ) where {VT, CT <: MBEChannelViewX1X{Q}} where {Q}

        return new{Q, VT, CT}(F0, ∇Sp, ∇Tp, ∇Da, ∇Dt, ∇Ma, ∇Mt)
    end
end




struct DSp end
struct MSp end
struct SSp end
struct TSp end

_spin_coeff(:: Type{pCh}, :: Type{pSp}, :: Type{SSp}) =  1/2
_spin_coeff(:: Type{pCh}, :: Type{pSp}, :: Type{TSp}) =  1/2
_spin_coeff(:: Type{pCh}, :: Type{xSp}, :: Type{SSp}) = -1/2
_spin_coeff(:: Type{pCh}, :: Type{xSp}, :: Type{TSp}) =  1/2
_spin_coeff(:: Type{pCh}, :: Type{dSp}, :: Type{SSp}) =  1/2
_spin_coeff(:: Type{pCh}, :: Type{dSp}, :: Type{TSp}) =  3/2

_spin_coeff(:: Type{aCh}, :: Type{pSp}, :: Type{DSp}) =  0.0
_spin_coeff(:: Type{aCh}, :: Type{pSp}, :: Type{MSp}) = -1.0
_spin_coeff(:: Type{aCh}, :: Type{xSp}, :: Type{DSp}) = -1/2
_spin_coeff(:: Type{aCh}, :: Type{xSp}, :: Type{MSp}) =  1/2
_spin_coeff(:: Type{aCh}, :: Type{dSp}, :: Type{DSp}) = -1/2
_spin_coeff(:: Type{aCh}, :: Type{dSp}, :: Type{MSp}) = -3/2

_spin_coeff(:: Type{tCh}, :: Type{pSp}, :: Type{DSp}) =  1/2
_spin_coeff(:: Type{tCh}, :: Type{pSp}, :: Type{MSp}) = -1/2
_spin_coeff(:: Type{tCh}, :: Type{xSp}, :: Type{DSp}) =  0.0
_spin_coeff(:: Type{tCh}, :: Type{xSp}, :: Type{MSp}) =  1.0
_spin_coeff(:: Type{tCh}, :: Type{dSp}, :: Type{DSp}) =  1.0
_spin_coeff(:: Type{tCh}, :: Type{dSp}, :: Type{MSp}) =  0.0


@inline function (F :: MBEVertexViewX1X{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{Sp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, Ch, Sp)
    end

    if γp
        # val += F.∇Sp(convert_frequency(Ω, ν, νp, Ch, pCh)...) * _spin_coeff(pCh, Sp, SSp)
        # val += F.∇Tp(convert_frequency(Ω, ν, νp, Ch, pCh)...) * _spin_coeff(pCh, Sp, TSp)
        if Sp === :pSp
            val += F.∇Sp(convert_frequency(Ω, ν, νp, Ch, pCh)...) * 1/2
        elseif Sp === :xSp
            val += F.∇Tp(convert_frequency(Ω, ν, Ω - νp, Ch, pCh)...) * -1/2
        elseif Sp === :dSp
            val += F.∇Sp(convert_frequency(Ω, ν, νp, Ch, pCh)...) * 1/2 * 2
            val += F.∇Tp(convert_frequency(Ω, ν, Ω - νp, Ch, pCh)...) * -1/2
        end
    end

    if γt
        val += F.∇Dt(convert_frequency(Ω, ν, νp, Ch, tCh)...) * _spin_coeff(tCh, Sp, DSp)
        val += F.∇Mt(convert_frequency(Ω, ν, νp, Ch, tCh)...) * _spin_coeff(tCh, Sp, MSp)
    end

    if γa
        val += F.∇Da(convert_frequency(Ω, ν, νp, Ch, aCh)...) * _spin_coeff(aCh, Sp, DSp)
        val += F.∇Ma(convert_frequency(Ω, ν, νp, Ch, aCh)...) * _spin_coeff(aCh, Sp, MSp)
    end

    return val
end



@inline function fixed_momentum_view(
    F  :: NL2_MBEVertex,
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch}
    ) where {Ch <: ChannelTag}

    F0 = fixed_momentum_view(F.F0, P, k, kp, Ch)

    ks_p = convert_momentum(P, k, kp, Ch, pCh)
    ks_t = convert_momentum(P, k, kp, Ch, tCh)
    ks_a = convert_momentum(P, k, kp, Ch, aCh)

    # ∇Sp = fixed_momentum_view(F.∇S, ks_p...)
    # ∇Tp = fixed_momentum_view(F.∇T, ks_p...)
    ∇Sp = fixed_momentum_view(F.∇P, ks_p...)
    ∇Tp = fixed_momentum_view(F.∇P, convert_momentum(P, k, P - kp, Ch, pCh)...)
    ∇Dt = fixed_momentum_view(F.∇D, ks_t...)
    ∇Mt = fixed_momentum_view(F.∇M, ks_t...)
    ∇Da = fixed_momentum_view(F.∇D, ks_a...)
    ∇Ma = fixed_momentum_view(F.∇M, ks_a...)

    MBEVertexViewX1X(F0, ∇Sp, ∇Tp, ∇Da, ∇Dt, ∇Ma, ∇Mt)
end

# -----------------------------------------------------------------------------
# mfRG update

function Base.:copy(
    ∇ :: NL2_MBEChannel{Q}
    ) :: NL2_MBEChannel{Q} where {Q}

    return NL2_MBEChannel(∇.U, copy(∇.η), copy(∇.λ), copy(∇.M))
end

function Base.:copy(
    F :: NL2_MBEVertex{Q}
    ) :: NL2_MBEVertex{Q} where {Q}

    return NL2_MBEVertex(copy(F.F0), copy(F.∇P), copy(F.∇D), copy(F.∇M))
end

function MatsubaraFunctions.set!(
    F :: NL2_MBEChannel,
    val :: Number,
    ) :: Nothing

    set!(F.η, val)
    set!(F.λ, val)
    set!(F.M, val)

    return nothing
end

function MatsubaraFunctions.set!(
    F :: NL2_MBEVertex,
    val :: Number,
    ) :: Nothing

    set!(F.∇P, val)
    set!(F.∇D, val)
    set!(F.∇M, val)

    return nothing
end


function MatsubaraFunctions.set!(
    ∇1 :: NL2_MBEChannel,
    ∇2 :: NL2_MBEChannel
    )  :: Nothing

    set!(∇1.η, ∇2.η)
    set!(∇1.λ, ∇2.λ)
    set!(∇1.M, ∇2.M)

    return nothing
end

function MatsubaraFunctions.set!(
    F1 :: NL2_MBEVertex,
    F2 :: NL2_MBEVertex
    )  :: Nothing

    set!(F1.∇P, F2.∇P)
    set!(F1.∇D, F2.∇D)
    set!(F1.∇M, F2.∇M)

    return nothing
end

function MatsubaraFunctions.add!(F0 :: NL2_MBEVertex, F :: NL2_Vertex)
    # F0 -> F0 + F
    @assert F0 === F.F0

    set!(F0, to_mbe(F))

    return nothing
end

function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    ∇     :: NL2_MBEChannel
    )     :: Nothing

    file[label * "/U"] = ∇.U
    MatsubaraFunctions.save!(file, label * "/η", ∇.η)
    MatsubaraFunctions.save!(file, label * "/λ", ∇.λ)
    MatsubaraFunctions.save!(file, label * "/M", ∇.M)

    return nothing
end

function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    F     :: NL2_MBEVertex
    )     :: Nothing

    MatsubaraFunctions.save!(file, label * "/F0", F.F0)
    MatsubaraFunctions.save!(file, label * "/∇P", F.∇P)
    MatsubaraFunctions.save!(file, label * "/∇D", F.∇D)
    MatsubaraFunctions.save!(file, label * "/∇M", F.∇M)

    return nothing
end


function load_vertex(
          :: Type{T},
    file  :: HDF5.File,
    label :: String
    )     :: T where {T <: NL2_MBEChannel}

    U = read(file, label * "/U")
    η = load_mesh_function(file, label * "/η")
    λ = load_mesh_function(file, label * "/λ")
    M = load_mesh_function(file, label * "/M")

    return T(U, η, λ, M)
end

function load_vertex(
          :: Type{T},
    file  :: HDF5.File,
    label :: String
    )     :: T where {T <: NL2_MBEVertex}

    if haskey(attributes(file[label * "/F0"]), "U")
        F0 = load_refvertex(file, label * "/F0")
    else
        try
            F0 = load_vertex(Vertex, file, label * "/F0")
        catch
            try
                F0 = load_vertex(NL_Vertex, file, label * "/F0")
            catch
                F0 = load_vertex(NL2_Vertex, file, label * "/F0")
            end
        end
    end
    ∇P = load_vertex(NL2_MBEChannel, file, label * "/∇P")
    ∇D = load_vertex(NL2_MBEChannel, file, label * "/∇D")
    ∇M = load_vertex(NL2_MBEChannel, file, label * "/∇M")

    return T(F0, ∇P, ∇D, ∇M)
end
