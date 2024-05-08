# 2-particle reducible vertex in the asymptotic decomposition

abstract type AbstractReducibleVertex{Q}; end
Base.eltype(::Type{<: AbstractReducibleVertex{Q}}) where {Q} = Q


struct Channel{Q <: Number} <: AbstractReducibleVertex{Q}
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

        @assert numK1 >= numK2[1] "No. bosonic frequencies in K1 must be larger than no. bosonic frequencies in K2"
        @assert numK1 >= numK2[2] "No. bosonic frequencies in K1 must be larger than no. fermionic frequencies in K2"
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
    γ :: AbstractReducibleVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(meshes(γ.K1, Val(1)))
end

function numK1(
    γ :: AbstractReducibleVertex
    ) :: Int64

    return N(meshes(γ.K1, Val(1)))
end

function numK2(
    γ :: AbstractReducibleVertex
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K2, Val(1))), N(meshes(γ.K2, Val(2)))
end

function numK3(
    γ :: AbstractReducibleVertex
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K3, Val(1))), N(meshes(γ.K3, Val(2)))
end

# setter methods
function MatsubaraFunctions.set!(
    γ1 :: AbstractReducibleVertex,
    γ2 :: AbstractReducibleVertex
    )  :: Nothing

    set!(γ1.K1, γ2.K1)
    set!(γ1.K2, γ2.K2)
    set!(γ1.K3, γ2.K3)

    return nothing
end

function MatsubaraFunctions.set!(
    γ1  :: AbstractReducibleVertex{Q},
    val :: Number,
    )   :: Nothing where {Q}

    set!(γ1.K1, Q(val))
    set!(γ1.K2, Q(val))
    set!(γ1.K3, Q(val))

    return nothing
end

# comparison
function Base.:(==)(
    γ1 :: AbstractReducibleVertex,
    γ2 :: AbstractReducibleVertex
    )  :: Bool
    return (γ1.K1 == γ2.K1) && (γ1.K2 == γ2.K2) && (γ1.K3 == γ2.K3)
end

# addition
function MatsubaraFunctions.add!(
    γ1 :: AbstractReducibleVertex,
    γ2 :: AbstractReducibleVertex
    )  :: Nothing

    add!(γ1.K1, γ2.K1)
    add!(γ1.K2, γ2.K2)
    add!(γ1.K3, γ2.K3)

    return nothing
end

function MatsubaraFunctions.mult_add!(
    γ1 :: AbstractReducibleVertex,
    γ2 :: AbstractReducibleVertex,
    val :: Number,
    )  :: Nothing

    mult_add!(γ1.K1, γ2.K1, val)
    mult_add!(γ1.K2, γ2.K2, val)
    mult_add!(γ1.K3, γ2.K3, val)

    return nothing
end

# length of channel
function Base.length(
    γ :: AbstractReducibleVertex
    ) :: Int64

    return length(γ.K1.data) + length(γ.K2.data) + length(γ.K3.data)
end

# maximum absolute value
function MatsubaraFunctions.absmax(
    γ :: AbstractReducibleVertex
    ) :: Float64

    return max(absmax(γ.K1), absmax(γ.K2), absmax(γ.K3))
end

# flatten into vector
function MatsubaraFunctions.flatten!(
    γ :: AbstractReducibleVertex,
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
    γ :: AbstractReducibleVertex{Q}
    ) :: Vector{Q} where {Q}

    x = Array{Q}(undef, length(γ))
    flatten!(γ, x)

    return x
end

# unflatten from vector
function MatsubaraFunctions.unflatten!(
    γ :: AbstractReducibleVertex,
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

# copy
function Base.:copy(
    γ :: Channel{Q}
    ) :: Channel{Q} where {Q}
    return Channel(copy(γ.K1), copy(γ.K2), copy(γ.K3))
end

# evaluator
@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    if is_inbounds(Ω, meshes(γ.K1, Val(1)))
        K1 && (val += γ.K1[Ω])

        if is_inbounds(Ω, meshes(γ.K2, Val(1)))
            ν1_inbounds = is_inbounds(ν, meshes(γ.K2, Val(2)))
            ν2_inbounds = is_inbounds(νp, meshes(γ.K2, Val(2)))

            if ν1_inbounds && ν2_inbounds
                K2 && (val += γ.K2[Ω, ν] + γ.K2[Ω, νp])
                K3 && (val += γ.K3(Ω, ν, νp))

            elseif ν1_inbounds
                K2 && (val += γ.K2[Ω, ν])

            elseif ν2_inbounds
                K2 && (val += γ.K2[Ω, νp])
            end
        end
    end

    return val
end


@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: MatsubaraFrequency
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case ν = ∞

    val = zero(Q)

    if K1
        if is_inbounds(Ω, meshes(γ.K1, Val(1)))
            val += γ.K1[Ω]

            if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(νp, meshes(γ.K2, Val(2)))
                val += γ.K2[Ω, νp]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(νp, meshes(γ.K2, Val(2)))
            val += γ.K2[Ω, νp]
        end
    end

    return val
end

@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case νp = ∞

    val = zero(Q)

    if K1
        if is_inbounds(Ω, meshes(γ.K1, Val(1)))
            K1 && (val += γ.K1[Ω])

            if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(ν, meshes(γ.K2, Val(2)))
                val += γ.K2[Ω, ν]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(ν, meshes(γ.K2, Val(2)))
            val += γ.K2[Ω, ν]
        end
    end

    return val
end

@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    # Implement special case ν = ∞ and νp = ∞

    val = zero(Q)

    if K1 && is_inbounds(Ω, meshes(γ.K1, Val(1)))
        val += γ.K1[Ω]
    end

    return val
end


# reducer
# Subtract the lower-order asymptotic vertices from the higher-order ones
function reduce!(
    γ :: Channel
    ) :: Nothing

    for iΩ in eachindex(meshes(γ.K2, Val(1)))
        Ω = value(meshes(γ.K2, Val(1))[iΩ])
        K1val = γ.K1[Ω]

        for iν in eachindex(meshes(γ.K2, Val(2)))
            ν = value(meshes(γ.K2, Val(2))[iν])
            γ.K2[Ω, ν] -= K1val
        end
    end

    for iΩ in eachindex(meshes(γ.K3, Val(1)))
        Ω = value(meshes(γ.K3, Val(1))[iΩ])
        K1val = γ.K1[Ω]

        for iν in eachindex(meshes(γ.K3, Val(2)))
            ν = value(meshes(γ.K3, Val(2))[iν])
            K2val = γ.K2[Ω, ν]

            for iνp in eachindex(meshes(γ.K3, Val(3)))
                νp = value(meshes(γ.K3, Val(3))[iνp])
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
    γ     :: AbstractReducibleVertex
    )     :: Nothing

    MatsubaraFunctions.save!(file, label * "/K1", γ.K1)
    MatsubaraFunctions.save!(file, label * "/K2", γ.K2)
    MatsubaraFunctions.save!(file, label * "/K3", γ.K3)

    return nothing
end

# load from HDF5
function load_channel(
          :: Type{T},
    file  :: HDF5.File,
    label :: String
    )     :: T where {T <: AbstractReducibleVertex}

    K1 = load_mesh_function(file, label * "/K1")
    K2 = load_mesh_function(file, label * "/K2")
    K3 = load_mesh_function(file, label * "/K3")

    return T(K1, K2, K3)
end
