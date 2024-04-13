# 2-particle reducible vertex in the asymptotic decomposition

struct NL_Channel{Q <: Number} <: AbstractReducibleVertex{Q}
    K1 :: NL_MF_K1{Q}
    K2 :: NL_MF_K2{Q}
    K3 :: NL_MF_K3{Q}

    function NL_Channel(
        K1 :: NL_MF_K1{Q},
        K2 :: NL_MF_K2{Q},
        K3 :: NL_MF_K3{Q},
        )  :: NL_Channel{Q} where {Q}

        return new{Q}(K1, K2, K3)
    end

    function NL_Channel(
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
              :: Type{Q} = ComplexF64,
        ) where {Q}

        mK1Ω = MatsubaraMesh(T, numK1, Boson)
        K1 = MeshFunction(mK1Ω, meshK; data_t = Q)
        set!(K1, 0)

        @assert numK1 > numK2[1] "No. bosonic frequencies in K1 must be larger than no. bosonic frequencies in K2"
        @assert numK1 > numK2[2] "No. bosonic frequencies in K1 must be larger than no. fermionic frequencies in K2"
        mK2Ω = MatsubaraMesh(T, numK2[1], Boson)
        mK2ν = MatsubaraMesh(T, numK2[2], Fermion)
        K2   = MeshFunction(mK2Ω, mK2ν, meshK; data_t = Q)
        set!(K2, 0)

        @assert all(numK2 .>= numK3) "Number of frequencies in K2 must be larger than in K3"
        mK3Ω = MatsubaraMesh(T, numK3[1], Boson)
        mK3ν = MatsubaraMesh(T, numK3[2], Fermion)
        K3   = MeshFunction(mK3Ω, mK3ν, mK3ν, meshK; data_t = Q)
        set!(K3, 0)

        return new{Q}(K1, K2, K3) :: NL_Channel{Q}
    end
end

# getter methods
function get_P_mesh(
    γ :: NL_Channel
    ) :: KMesh

    return meshes(γ.K1, Val(2))
end

function numP(
    γ :: NL_Channel
    ) :: Int64

    return length(meshes(γ.K1, Val(2)))
end

# copy
function Base.:copy(
    γ :: NL_Channel{Q}
    ) :: NL_Channel{Q} where {Q}
    return NL_Channel(copy(γ.K1), copy(γ.K2), copy(γ.K3))
end


# evaluator
@inline function (γ :: NL_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # We have only bosonic momentum dependence, so drop k and kp arguments.

    return γ(Ω, ν, νp, P_; K1, K2, K3)
end


@inline function (γ :: NL_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))

    if is_inbounds(Ω, meshes(γ.K1, Val(1)))
        K1 && (val += γ.K1[Ω, P])

        if is_inbounds(Ω, meshes(γ.K2, Val(1)))
            ν1_inbounds = is_inbounds(ν, meshes(γ.K2, Val(2)))
            ν2_inbounds = is_inbounds(νp, meshes(γ.K2, Val(2)))

            if ν1_inbounds && ν2_inbounds
                K2 && (val += γ.K2[Ω, ν, P] + γ.K2[Ω, νp, P])

                if (K3 && is_inbounds( Ω, meshes(γ.K3, Val(1))) &&
                          is_inbounds( ν, meshes(γ.K3, Val(2))) &&
                          is_inbounds(νp, meshes(γ.K3, Val(3))))
                    val += γ.K3[Ω, ν, νp, P]
                end

            elseif ν1_inbounds
                K2 && (val += γ.K2[Ω, ν, P])

            elseif ν2_inbounds
                K2 && (val += γ.K2[Ω, νp, P])
            end
        end
    end

    return val
end


@inline function (γ :: NL_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case ν = ∞

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))

    if K1
        if is_inbounds(Ω, meshes(γ.K1, Val(1)))
            val += γ.K1[Ω, P]

            if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(νp, meshes(γ.K2, Val(2)))
                val += γ.K2[Ω, νp, P]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(νp, meshes(γ.K2, Val(2)))
            val += γ.K2[Ω, νp, P]
        end
    end

    return val
end

@inline function (γ :: NL_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency,
    P_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case νp = ∞

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))

    if K1
        if is_inbounds(Ω, meshes(γ.K1, Val(1)))
            K1 && (val += γ.K1[Ω, P])

            if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(ν, meshes(γ.K2, Val(2)))
                val += γ.K2[Ω, ν, P]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(ν, meshes(γ.K2, Val(2)))
            val += γ.K2[Ω, ν, P]
        end
    end

    return val
end

@inline function (γ :: NL_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency,
    P_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case ν = ∞ and νp = ∞

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))

    if K1 && is_inbounds(Ω, meshes(γ.K1, Val(1)))
        val += γ.K1[Ω, P]
    end

    return val
end


# Evaluation of local vertices with auxiliary momentum arguments
@inline function (γ :: Union{Channel{Q}, RefVertex{Q}, Vertex{Q}})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
    args...
    ;
    kwargs...
    )  :: Q where {Q}

    return γ(Ω, ν, νp, args...; kwargs...)
end


# reducer
# Subtract the lower-order asymptotic vertices from the higher-order ones
function reduce!(
    γ :: NL_Channel,
    ) :: Nothing

    for iP in eachindex(get_P_mesh(γ))
        P = value(get_P_mesh(γ)[iP])

        for iΩ in eachindex(meshes(γ.K2, Val(1)))
            Ω = value(meshes(γ.K2, Val(1))[iΩ])
            K1val = γ.K1[Ω, P]

            for iν in eachindex(meshes(γ.K2, Val(2)))
                ν = value(meshes(γ.K2, Val(2))[iν])
                γ.K2[Ω, ν, P] -= K1val
            end
        end

        for iΩ in eachindex(meshes(γ.K3, Val(1)))
            Ω = value(meshes(γ.K3, Val(1))[iΩ])
            K1val = γ.K1[Ω, P]

            for iν in eachindex(meshes(γ.K3, Val(2)))
                ν = value(meshes(γ.K3, Val(2))[iν])
                K2val = γ.K2[Ω, ν, P]

                for iνp in eachindex(meshes(γ.K3, Val(3)))
                    νp = value(meshes(γ.K3, Val(3))[iνp])
                    γ.K3[Ω, ν, νp, P] -= K1val + K2val + γ.K2[Ω, νp, P]
                end
            end
        end

    end

    return nothing
end
