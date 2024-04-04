# 2-particle reducible vertex in the asymptotic decomposition

struct Channel{Q <: Number}
    K1 :: MF_K1{Q}
    K2 :: MF_K2{Q}
    K3 :: MF_K3{Q}

    function Channel(
        K1 :: MF_K1{Q},
        K2 :: MF_K2{Q},
        K3 :: MF_K3{Q},
        )  :: Channel{Q} where {Q}

        return new{Q}(K1, K2, K3)
    end

    function Channel(
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
              :: Type{Q} = ComplexF64,
        ) where {Q}

        mK1Ω = MatsubaraMesh(T, numK1, Boson)
        K1 = MeshFunction(mK1Ω; data_t = Q)
        set!(K1, 0)

        @assert numK1 > numK2[1] "No. bosonic frequencies in K1 must be larger than no. bosonic frequencies in K2"
        @assert numK1 > numK2[2] "No. bosonic frequencies in K1 must be larger than no. fermionic frequencies in K2"
        mK2Ω = MatsubaraMesh(T, numK2[1], Boson)
        mK2ν = MatsubaraMesh(T, numK2[2], Fermion)
        K2   = MeshFunction(mK2Ω, mK2ν; data_t = Q)
        set!(K2, 0)

        @assert all(numK2 .>= numK3) "Number of frequencies in K2 must be larger than in K3"
        mK3Ω = MatsubaraMesh(T, numK3[1], Boson)
        mK3ν = MatsubaraMesh(T, numK3[2], Fermion)
        K3   = MeshFunction(mK3Ω, mK3ν, mK3ν; data_t = Q)
        set!(K3, 0)

        return new{Q}(K1, K2, K3) :: Channel{Q}
    end
end

# getter methods
function MatsubaraFunctions.temperature(
    γ :: Channel
    ) :: Float64

    return MatsubaraFunctions.temperature(meshes(γ.K1, 1))
end

function numK1(
    γ :: Channel
    ) :: Int64

    return N(meshes(γ.K1, 1))
end

function numK2(
    γ :: Channel
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K2, 1)), N(meshes(γ.K2, 2))
end

function numK3(
    γ :: Channel
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K3, 1)), N(meshes(γ.K3, 2))
end

# setter methods
function MatsubaraFunctions.set!(
    γ1 :: Channel,
    γ2 :: Channel
    )  :: Nothing

    set!(γ1.K1, γ2.K1)
    set!(γ1.K2, γ2.K2)
    set!(γ1.K3, γ2.K3)

    return nothing
end

function reset!(
    γ :: Channel
    ) :: Nothing

    set!(γ.K1, 0)
    set!(γ.K2, 0)
    set!(γ.K3, 0)

    return nothing
end

# comparison
function Base.:(==)(
    γ1 :: Channel,
    γ2 :: Channel
    )  :: Bool
    return (γ1.K1 == γ2.K1) && (γ1.K2 == γ2.K2) && (γ1.K3 == γ2.K3)
end

# copy
function Base.:copy(
    γ :: Channel
    ) :: Channel

    return Channel(copy(γ.K1), copy(γ.K2), copy(γ.K3))
end

# addition
function MatsubaraFunctions.add!(
    γ1 :: Channel,
    γ2 :: Channel
    )  :: Nothing

    add!(γ1.K1, γ2.K1)
    add!(γ1.K2, γ2.K2)
    add!(γ1.K3, γ2.K3)

    return nothing
end

# length of channel
function Base.length(
    γ :: Channel
    ) :: Int64

    return length(γ.K1.data) + length(γ.K2.data) + length(γ.K3.data)
end

# maximum absolute value
function MatsubaraFunctions.absmax(
    γ :: Channel
    ) :: Float64

    return max(absmax(γ.K1), absmax(γ.K2), absmax(γ.K3))
end

# flatten into vector
function MatsubaraFunctions.flatten!(
    γ :: Channel,
    x :: AbstractVector
    ) :: Nothing

    offset = 0
    lenK1  = length(γ.K1.data)
    lenK2  = length(γ.K2.data)
    lenK3  = length(γ.K3.data)

    flatten!(γ.K1, @view x[1 + offset : offset + lenK1])
    offset += lenK1

    flatten!(γ.K2, @view x[1 + offset : offset + lenK2])
    offset += lenK2

    flatten!(γ.K3, @view x[1 + offset : offset + lenK3])
    offset += lenK3

    @assert offset == length(x) "Dimension mismatch between channel and target vector"
    return nothing
end

function MatsubaraFunctions.flatten(
    γ :: Channel{Q}
    ) :: Vector{Q} where {Q}

    x = Array{Q}(undef, length(γ))
    flatten!(γ, x)

    return x
end

# unflatten from vector
function MatsubaraFunctions.unflatten!(
    γ :: Channel,
    x :: AbstractVector
    ) :: Nothing

    offset = 0
    lenK1  = length(γ.K1.data)
    lenK2  = length(γ.K2.data)
    lenK3  = length(γ.K3.data)

    unflatten!(γ.K1, @view x[1 + offset : offset + lenK1])
    offset += lenK1

    unflatten!(γ.K2, @view x[1 + offset : offset + lenK2])
    offset += lenK2

    unflatten!(γ.K3, @view x[1 + offset : offset + lenK3])
    offset += lenK3

    @assert offset == length(x) "Dimension mismatch between channel and target vector"
    return nothing
end

# evaluator
# @inline function box_eval(
#     f :: MeshFunction{DD, Q},
#     w :: Vararg{MatsubaraFrequency, DD}
#     ) :: Q where {DD, Q}

#     if any(ntuple(i -> !is_inbounds(w[i], meshes(f, i)), DD))
#         return zero(Q)
#     else
#         return f[w...]
#     end
# end

@inline function box_eval(
    f  :: MeshFunction{1, Q},
    w1 :: MatsubaraFrequency,
    ) :: Q where {Q}

    is_inbounds(w1, meshes(f, 1)) || return zero(Q)
    return f[w1]
end

@inline function box_eval(
    f  :: MeshFunction{2, Q},
    w1 :: MatsubaraFrequency,
    w2 :: MatsubaraFrequency,
    ) :: Q where {Q}

    is_inbounds(w1, meshes(f, 1)) || return zero(Q)
    is_inbounds(w2, meshes(f, 2)) || return zero(Q)
    return f[w1, w2]
end


@inline function box_eval(
    f  :: MeshFunction{3, Q},
    w1 :: MatsubaraFrequency,
    w2 :: MatsubaraFrequency,
    w3 :: MatsubaraFrequency,
    ) :: Q where {Q}

    is_inbounds(w1, meshes(f, 1)) || return zero(Q)
    is_inbounds(w2, meshes(f, 2)) || return zero(Q)
    is_inbounds(w3, meshes(f, 3)) || return zero(Q)
    return f[w1, w2, w3]
end


@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency
    )  :: Q where {Q}

    val = zero(Q)

    if is_inbounds(Ω, meshes(γ.K1, 1))
        val += γ.K1[Ω]

        if is_inbounds(Ω, meshes(γ.K2, 1))
            ν1_inbounds = is_inbounds(ν, meshes(γ.K2, 2))
            ν2_inbounds = is_inbounds(νp, meshes(γ.K2, 2))

            if ν1_inbounds && ν2_inbounds
                val += γ.K2[Ω, ν] + γ.K2[Ω, νp] + box_eval(γ.K3, Ω, ν, νp)

            elseif ν1_inbounds
                val += γ.K2[Ω, ν]

            elseif ν2_inbounds
                val += γ.K2[Ω, νp]
            end
        end
    end

    return val
end


@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: MatsubaraFrequency
    )  :: Q where {Q}

    # Implement special case ν = ∞

    val = zero(Q)

    if is_inbounds(Ω, meshes(γ.K1, 1))
        val += γ.K1[Ω]

        if is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(νp, meshes(γ.K2, 2))
            val += γ.K2[Ω, νp]
        end
    end

    return val
end

@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency
    )  :: Q where {Q}

    # Implement special case νp = ∞

    val = zero(Q)

    if is_inbounds(Ω, meshes(γ.K1, 1))
        val += γ.K1[Ω]

        if is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(ν, meshes(γ.K2, 2))
            val += γ.K2[Ω, ν]
        end
    end

    return val
end

@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency
    )  :: Q where {Q}

    # Implement special case ν = ∞ and νp = ∞

    val = zero(Q)

    if is_inbounds(Ω, meshes(γ.K1, 1))
        val += γ.K1[Ω]
    end

    return val
end


# reducer
# Subtract the lower-order asymptotic vertices from the higher-order ones
function reduce!(
    γ :: Channel
    ) :: Nothing

    for iΩ in eachindex(meshes(γ.K2, 1))
        Ω = value(meshes(γ.K2, 1)[iΩ])
        K1val = γ.K1[Ω]

        for iν in eachindex(meshes(γ.K2, 2))
            ν = value(meshes(γ.K2, 2)[iν])
            γ.K2[Ω, ν] -= K1val
        end
    end

    for iΩ in eachindex(meshes(γ.K3, 1))
        Ω = value(meshes(γ.K3, 1)[iΩ])
        K1val = γ.K1[Ω]

        for iν in eachindex(meshes(γ.K3, 2))
            ν = value(meshes(γ.K3, 2)[iν])
            K2val = γ.K2[Ω, ν]

            for iνp in eachindex(meshes(γ.K3, 3))
                νp = value(meshes(γ.K3, 3)[iνp])
                γ.K3[Ω, ν, νp] -= K1val + K2val + γ.K2[Ω, νp]
            end
        end
    end

    return nothing
end

# save to HDF5
function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    γ     :: Channel
    )     :: Nothing

    MatsubaraFunctions.save!(file, label * "/K1", γ.K1)
    MatsubaraFunctions.save!(file, label * "/K2", γ.K2)
    MatsubaraFunctions.save!(file, label * "/K3", γ.K3)

    return nothing
end

# load from HDF5
function load_channel(
    file  :: HDF5.File,
    label :: String
    )     :: Channel

    K1 = load_mesh_function(file, label * "/K1")
    K2 = load_mesh_function(file, label * "/K2")
    K3 = load_mesh_function(file, label * "/K3")

    return Channel(K1, K2, K3)
end
