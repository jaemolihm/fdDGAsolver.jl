# 2-particle reducible vertex in the asymptotic decomposition

const FMesh = Mesh{MeshPoint{MatsubaraFrequency{Fermion}}, MatsubaraDomain}
const BMesh = Mesh{MeshPoint{MatsubaraFrequency{Boson}}, MatsubaraDomain}
const MF_K1{Q} = MeshFunction{1, Q, Tuple{BMesh}, Array{Q, 1}}
const MF_K2{Q} = MeshFunction{2, Q, Tuple{FMesh, BMesh}, Array{Q, 2}}
const MF_K3{Q} = MeshFunction{3, Q, Tuple{FMesh, FMesh, BMesh}, Array{Q, 3}}

struct ReducibleVertex{Q <: Number}
    channel :: Symbol
    K1 :: MF_K1{Q}
    K2 :: MF_K2{Q}
    K3 :: MF_K3{Q}

    function ReducibleVertex(
        channel :: Symbol,
        K1 :: MF_K1{Q},
        K2 :: MF_K2{Q},
        K3 :: MF_K3{Q},
        )  where {Q}

        return new{Q}(channel, K1, K2, K3)
    end

    function ReducibleVertex(
        T :: Float64,
        channel :: Symbol,
        numK1 :: Int,
        numK2 :: NTuple{2, Int},
        numK3 :: NTuple{2, Int},
        :: Type{Q} = ComplexF64,
        ) where {Q}

        @assert channel ∈ (:a, :p, :t) "channel must be :a, :p, or :t"

        mK1v = MatsubaraMesh(T, numK1, Boson)
        K1 = MeshFunction(mK1v; data_t = Q)
        set!(K1, 0)

        @assert numK1 > numK2[1] "No. bosonic frequencies in K1 must be larger than no. fermionic frequencies in K2"
        @assert numK1 > numK2[2] "No. bosonic frequencies in K1 must be larger than no. bosonic frequencies in K2"
        gK2v = MatsubaraMesh(T, numK2[1], Fermion)
        gK2w = MatsubaraMesh(T, numK2[2], Boson)
        K2   = MeshFunction(gK2v, gK2w; data_t = Q)
        set!(K2, 0)

        @assert numK2 > numK3 "Number of frequencies in K2 must be larger than in K3"
        gK3v = MatsubaraMesh(T, numK3[1], Fermion)
        gK3w = MatsubaraMesh(T, numK3[2], Boson)
        K3   = MeshFunction(gK3v, gK3v, gK3w; data_t = Q)
        set!(K3, 0)

        return new{Q}(channel, K1, K2, K3) :: ReducibleVertex{Q}
    end
end

# getter methods
function MatsubaraFunctions.temperature(
    γ :: ReducibleVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(meshes(γ.K1, 1))
end

function numK1(
    γ :: ReducibleVertex
    ) :: Int64

    return N(meshes(γ.K1, 1))
end

function numK2(
    γ :: ReducibleVertex
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K2, 1)), N(meshes(γ.K2, 2))
end

function numK3(
    γ :: ReducibleVertex
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K3, 1)), N(meshes(γ.K3, 2))
end

# setter methods
function MatsubaraFunctions.set!(
    γ1 :: ReducibleVertex,
    γ2 :: ReducibleVertex
    )  :: Nothing

    set!(γ1.K1, γ2.K1)
    set!(γ1.K2, γ2.K2)
    set!(γ1.K3, γ2.K3)

    return nothing
end

# comparison
function Base.:(==)(
    γ1 :: ReducibleVertex,
    γ2 :: ReducibleVertex
    )  :: Bool
    return (γ1.channel == γ2.channel) && (γ1.K1 == γ2.K1) && (γ1.K2 == γ2.K2) && (γ1.K3 == γ2.K3)
end

# copy
function Base.:copy(
    γ :: ReducibleVertex
    ) :: ReducibleVertex

    return ReducibleVertex(γ.channel, copy(γ.K1), copy(γ.K2), copy(γ.K3))
end

# addition
function MatsubaraFunctions.add!(
    γ1 :: ReducibleVertex,
    γ2 :: ReducibleVertex
    )  :: Nothing

    add!(γ1.K1, γ2.K1)
    add!(γ1.K2, γ2.K2)
    add!(γ1.K3, γ2.K3)

    return nothing
end

# length of ReducibleVertex
function Base.length(
    γ :: ReducibleVertex
    ) :: Int64

    return length(γ.K1.data) + length(γ.K2.data) + length(γ.K3.data)
end

# maximum absolute value
function MatsubaraFunctions.absmax(
    γ :: ReducibleVertex
    ) :: Float64

    return max(absmax(γ.K1), absmax(γ.K2), absmax(γ.K3))
end

# flatten into vector
function MatsubaraFunctions.flatten!(
    γ :: ReducibleVertex,
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

    @assert offset == length(x) "Dimension mismatch between ReducibleVertex and target vector"
    return nothing
end

function MatsubaraFunctions.flatten(
    γ :: ReducibleVertex{Q}
    ) :: Vector{Q} where {Q}

    x = Array{Q}(undef, length(γ))
    flatten!(γ, x)

    return x
end

# unflatten from vector
function MatsubaraFunctions.unflatten!(
    γ :: ReducibleVertex,
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

    @assert offset == length(x) "Dimension mismatch between ReducibleVertex and target vector"
    return nothing
end

# evaluator
@inline function box_eval(
    f :: MeshFunction{DD, Q},
    w :: Vararg{MatsubaraFrequency, DD}
    ) :: Q where {DD, Q}

    if any(ntuple(i -> !is_inbounds(w[i], meshes(f, i)), DD))
        return zero(Q)
    else
        return f[w...]
    end
end

@inline function (γ :: ReducibleVertex{Q})(
    v1 :: MatsubaraFrequency,
    v2 :: MatsubaraFrequency,
    w  :: MatsubaraFrequency,
    )  :: Q where {Q}

    val = zero(Q)

    if is_inbounds(w, meshes(γ.K1, 1))
        val += γ.K1[w]

        if is_inbounds(w, meshes(γ.K2, 2))
            v1_inbounds = is_inbounds(v1, meshes(γ.K2, 1))
            v2_inbounds = is_inbounds(v2, meshes(γ.K2, 1))

            if v1_inbounds && v2_inbounds
                val += γ.K2[v1, w] + γ.K2[v2, w] + box_eval(γ.K3, v1, v2, w)

            elseif v1_inbounds
                val += γ.K2[v1, w]

            elseif v2_inbounds
                val += γ.K2[v2, w]
            end
        end
    end

    return val
end

# reducer
# Subtract the lower-order asymptotic vertices from the higher-order ones
function reduce!(
    γ :: ReducibleVertex
    ) :: Nothing

    for w in value.(meshes(γ.K2, 2))
        K1val = γ.K1[w]

        for v in value.(meshes(γ.K2, 1))
            γ.K2[v, w] -= K1val
        end
    end

    for w in value.(meshes(γ.K3, 3))
        K1val = γ.K1[w]

        for vp in value.(meshes(γ.K3, 2))
            K2pval = γ.K2[vp, w]

            for v in value.(meshes(γ.K3, 1))
                K2val = γ.K2[v, w]
                γ.K3[v, vp, w] -= K1val + K2pval + K2val
            end
        end
    end

    return nothing
end

# save to HDF5
function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    γ     :: ReducibleVertex
    )     :: Nothing

    grp = create_group(file, label)

    attributes(grp)["channel"] = String(γ.channel)

    MatsubaraFunctions.save!(file, label * "/K1", γ.K1)
    MatsubaraFunctions.save!(file, label * "/K2", γ.K2)
    MatsubaraFunctions.save!(file, label * "/K3", γ.K3)

    return nothing
end

# load from HDF5
function load_reducible_vertex(
    file  :: HDF5.File,
    label :: String
    )     :: ReducibleVertex

    channel = Symbol(read_attribute(file[label], "channel"))

    K1 = load_mesh_function(file, label * "/K1")
    K2 = load_mesh_function(file, label * "/K2")
    K3 = load_mesh_function(file, label * "/K3")

    return ReducibleVertex(channel, K1, K2, K3)
end
