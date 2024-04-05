# 2-particle reducible vertex in the asymptotic decomposition

struct NL_Channel{Q <: Number}
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
function MatsubaraFunctions.temperature(
    γ :: NL_Channel
    ) :: Float64

    return MatsubaraFunctions.temperature(meshes(γ.K1, 1))
end

function get_P_mesh(
    γ :: NL_Channel
    ) :: KMesh

    return meshes(γ.K1, 2)
end

function numK1(
    γ :: NL_Channel
    ) :: Int64

    return N(meshes(γ.K1, 1))
end

function numK2(
    γ :: NL_Channel
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K2, 1)), N(meshes(γ.K2, 2))
end

function numK3(
    γ :: NL_Channel
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K3, 1)), N(meshes(γ.K3, 2))
end

function numP(
    γ :: NL_Channel
    ) :: Int64

    return length(meshes(γ.K1, 2))
end

# setter methods
function MatsubaraFunctions.set!(
    γ1 :: NL_Channel,
    γ2 :: NL_Channel
    )  :: Nothing

    set!(γ1.K1, γ2.K1)
    set!(γ1.K2, γ2.K2)
    set!(γ1.K3, γ2.K3)

    return nothing
end

function MatsubaraFunctions.set!(
    γ1  :: NL_Channel{Q},
    val :: Number
    )   :: Nothing where {Q}

    set!(γ1.K1, Q(val))
    set!(γ1.K2, Q(val))
    set!(γ1.K3, Q(val))

    return nothing
end

# comparison
function Base.:(==)(
    γ1 :: NL_Channel,
    γ2 :: NL_Channel
    )  :: Bool
    return (γ1.K1 == γ2.K1) && (γ1.K2 == γ2.K2) && (γ1.K3 == γ2.K3)
end

# copy
function Base.:copy(
    γ :: NL_Channel
    ) :: NL_Channel

    return NL_Channel(copy(γ.K1), copy(γ.K2), copy(γ.K3))
end

# addition
function MatsubaraFunctions.add!(
    γ1 :: NL_Channel,
    γ2 :: NL_Channel
    )  :: Nothing

    add!(γ1.K1, γ2.K1)
    add!(γ1.K2, γ2.K2)
    add!(γ1.K3, γ2.K3)

    return nothing
end

# length of channel
function Base.length(
    γ :: NL_Channel
    ) :: Int64

    return length(γ.K1.data) + length(γ.K2.data) + length(γ.K3.data)
end

# maximum absolute value
function MatsubaraFunctions.absmax(
    γ :: NL_Channel
    ) :: Float64

    return max(absmax(γ.K1), absmax(γ.K2), absmax(γ.K3))
end

# flatten into vector
function MatsubaraFunctions.flatten!(
    γ :: NL_Channel,
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
    γ :: NL_Channel{Q}
    ) :: Vector{Q} where {Q}

    x = Array{Q}(undef, length(γ))
    flatten!(γ, x)

    return x
end

# unflatten from vector
function MatsubaraFunctions.unflatten!(
    γ :: NL_Channel,
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
@inline function (γ :: NL_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P_ :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
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
    P_ :: BrillouinPoint,
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))

    if is_inbounds(Ω, meshes(γ.K1, 1))
        K1 && (val += γ.K1[Ω, P])

        if is_inbounds(Ω, meshes(γ.K2, 1))
            ν1_inbounds = is_inbounds(ν, meshes(γ.K2, 2))
            ν2_inbounds = is_inbounds(νp, meshes(γ.K2, 2))

            if ν1_inbounds && ν2_inbounds
                K2 && (val += γ.K2[Ω, ν, P] + γ.K2[Ω, νp, P])
                K3 && (val += box_eval(γ.K3, Ω, ν, νp, P))

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
    P_ :: BrillouinPoint,
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case ν = ∞

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))

    if K1
        if is_inbounds(Ω, meshes(γ.K1, 1))
            val += γ.K1[Ω, P]

            if K2 && is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(νp, meshes(γ.K2, 2))
                val += γ.K2[Ω, νp, P]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(νp, meshes(γ.K2, 2))
            val += γ.K2[Ω, νp, P]
        end
    end

    return val
end

@inline function (γ :: NL_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency,
    P_ :: BrillouinPoint,
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case νp = ∞

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))

    if K1
        if is_inbounds(Ω, meshes(γ.K1, 1))
            K1 && (val += γ.K1[Ω, P])

            if K2 && is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(ν, meshes(γ.K2, 2))
                val += γ.K2[Ω, ν, P]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, 1)) && is_inbounds(ν, meshes(γ.K2, 2))
            val += γ.K2[Ω, ν, P]
        end
    end

    return val
end

@inline function (γ :: NL_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency,
    P_ :: BrillouinPoint,
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case ν = ∞ and νp = ∞

    val = zero(Q)

    P = fold_back(P_, get_P_mesh(γ))

    if K1 && is_inbounds(Ω, meshes(γ.K1, 1))
        val += γ.K1[Ω, P]
    end

    return val
end


# Evaluation of local vertices with auxiliary momentum arguments
@inline function (γ :: Union{Channel{Q}, RefVertex{Q}, Vertex{Q}})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
    args...
    ;
    kwargs...
    )  :: Q where {Q}

    return γ(Ω, ν, νp, args...; kwargs...)
end



# reducer
# Subtract the lower-order asymptotic vertices from the higher-order ones
function reduce!(
    γ :: NL_Channel
    ) :: Nothing

    for iP in eachindex(get_P_mesh(γ))
        P = value(get_P_mesh(γ)[iP])

        for iΩ in eachindex(meshes(γ.K2, 1))
            Ω = value(meshes(γ.K2, 1)[iΩ])
            K1val = γ.K1[Ω, P]

            for iν in eachindex(meshes(γ.K2, 2))
                ν = value(meshes(γ.K2, 2)[iν])
                γ.K2[Ω, ν, P] -= K1val
            end
        end

        for iΩ in eachindex(meshes(γ.K3, 1))
            Ω = value(meshes(γ.K3, 1)[iΩ])
            K1val = γ.K1[Ω, P]

            for iν in eachindex(meshes(γ.K3, 2))
                ν = value(meshes(γ.K3, 2)[iν])
                K2val = γ.K2[Ω, ν, P]

                for iνp in eachindex(meshes(γ.K3, 3))
                    νp = value(meshes(γ.K3, 3)[iνp])
                    γ.K3[Ω, ν, νp, P] -= K1val + K2val + γ.K2[Ω, νp, P]
                end
            end
        end

    end

    return nothing
end

# save to HDF5
function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    γ     :: NL_Channel
    )     :: Nothing

    MatsubaraFunctions.save!(file, label * "/K1", γ.K1)
    MatsubaraFunctions.save!(file, label * "/K2", γ.K2)
    MatsubaraFunctions.save!(file, label * "/K3", γ.K3)

    return nothing
end

# load from HDF5
function load_nonlocal_channel(
    file  :: HDF5.File,
    label :: String
    )     :: NL_Channel

    K1 = load_mesh_function(file, label * "/K1")
    K2 = load_mesh_function(file, label * "/K2")
    K3 = load_mesh_function(file, label * "/K3")

    return NL_Channel(K1, K2, K3)
end
