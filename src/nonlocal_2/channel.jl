# 2-particle reducible vertex in the asymptotic decomposition

struct NL2_Channel{Q <: Number} <: AbstractReducibleVertex{Q}
    K1 :: NL_MF_K1{Q}
    K2 :: NL2_MF_K2{Q}
    K3 :: NL_MF_K3{Q}

    function NL2_Channel(
        K1 :: NL_MF_K1{Q},
        K2 :: NL2_MF_K2{Q},
        K3 :: NL_MF_K3{Q},
        )  :: NL2_Channel{Q} where {Q}

        return new{Q}(K1, K2, K3)
    end

    function NL2_Channel(
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
        K2   = MeshFunction(mK2Ω, mK2ν, meshK, meshK; data_t = Q)
        set!(K2, 0)

        @assert all(numK2 .>= numK3) "Number of frequencies in K2 must be larger than in K3"
        mK3Ω = MatsubaraMesh(T, numK3[1], Boson)
        mK3ν = MatsubaraMesh(T, numK3[2], Fermion)
        K3   = MeshFunction(mK3Ω, mK3ν, mK3ν, meshK; data_t = Q)
        set!(K3, 0)

        return new{Q}(K1, K2, K3) :: NL2_Channel{Q}
    end
end

# getter methods
function get_P_mesh(
    γ :: NL2_Channel
    ) :: KMesh

    return meshes(γ.K1, 2)
end

function numP(
    γ :: NL2_Channel
    ) :: Int64

    return length(meshes(γ.K1, 2))
end

# copy
function Base.:copy(
    γ :: NL2_Channel{Q}
    ) :: NL2_Channel{Q} where {Q}
    return NL2_Channel(copy(γ.K1), copy(γ.K2), copy(γ.K3))
end


# evaluator
@inline function (γ :: NL2_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    q_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))
    k = fold_back(k_, get_P_mesh(γ))
    q = fold_back(q_, get_P_mesh(γ))

    if is_inbounds(Ω, meshes(γ.K1, 1))
        K1 && (val += γ.K1[Ω, P])

        if is_inbounds(Ω, meshes(γ.K2, 1))
            ν1_inbounds = is_inbounds(ν, meshes(γ.K2, 2))
            ν2_inbounds = is_inbounds(ω, meshes(γ.K2, 2))

            if ν1_inbounds && ν2_inbounds
                K2 && (val += γ.K2[Ω, ν, P, k] + γ.K2[Ω, ω, P, q])

                if (K3 && is_inbounds(Ω, meshes(γ.K3, 1)) &&
                          is_inbounds(ν, meshes(γ.K3, 2)) &&
                          is_inbounds(ω, meshes(γ.K3, 3)))
                    val += γ.K3[Ω, ν, ω, P]
                end

            elseif ν1_inbounds
                K2 && (val += γ.K2[Ω, ν, P, k])

            elseif ν2_inbounds
                K2 && (val += γ.K2[Ω, ω, P, q])
            end
        end
    end

    return val
end


@inline function (γ :: NL2_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    ω  :: MatsubaraFrequency,
    P_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    q_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case ν = ∞

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))
    q = fold_back(q_, get_P_mesh(γ))

    if K1
        if is_inbounds(Ω, meshes(γ.K1, 1))
            val += γ.K1[Ω, P]

            if K2 && is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(ω, meshes(γ.K2, 2))
                val += γ.K2[Ω, ω, P, q]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(ω, meshes(γ.K2, 2))
            val += γ.K2[Ω, ω, P, q]
        end
    end

    return val
end

@inline function (γ :: NL2_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    ω  :: InfiniteMatsubaraFrequency,
    P_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    q_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case ω = ∞

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))
    k = fold_back(k_, get_P_mesh(γ))

    if K1
        if is_inbounds(Ω, meshes(γ.K1, 1))
            K1 && (val += γ.K1[Ω, P])

            if K2 && is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(ν, meshes(γ.K2, 2))
                val += γ.K2[Ω, ν, P, k]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(ν, meshes(γ.K2, 2))
            val += γ.K2[Ω, ν, P, k]
        end
    end

    return val
end

@inline function (γ :: NL2_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    ω :: InfiniteMatsubaraFrequency,
    P_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    q_ :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case ν = ∞ and ω = ∞

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))

    if K1 && is_inbounds(Ω, meshes(γ.K1, 1))
        val += γ.K1[Ω, P]
    end

    return val
end


# reducer
# Subtract the lower-order asymptotic vertices from the higher-order ones
function reduce!(
    γ :: NL2_Channel,
    ;
    max_class :: Int = 3,
    ) :: Nothing

    # max_class: asymptotic classes higher than max_class is ignored.

    max_class == 1 && return

    for iP in eachindex(get_P_mesh(γ))
        P = value(get_P_mesh(γ)[iP])

        if max_class >= 2
            for iΩ in eachindex(meshes(γ.K2, 1))
                Ω = value(meshes(γ.K2, 1)[iΩ])
                view(γ.K2, Ω, :, P, :) .-= γ.K1[Ω, P]
            end
        end

        if max_class >= 3
            for iΩ in eachindex(meshes(γ.K3, 1))
                Ω = value(meshes(γ.K3, 1)[iΩ])
                view(γ.K3, Ω, :, :, P) .-= γ.K1[Ω, P]

                for iν in eachindex(meshes(γ.K3, 2))
                    ν = value(meshes(γ.K3, 2)[iν])
                    view(γ.K3, Ω, ν, :, P) .-= γ.K2[Ω, ν, P, kSW]
                end

                for iω in eachindex(meshes(γ.K3, 3))
                    ω = value(meshes(γ.K3, 3)[iω])
                    view(γ.K3, Ω, :, ω, P) .-= γ.K2[Ω, ω, P, kSW]
                end
            end
        end

    end

    return nothing
end
