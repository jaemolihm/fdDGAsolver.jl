function MatsubaraFunctions.add!(f1 :: MeshFunction, f2 :: MeshFunction, α :: Number) :: Nothing
    MatsubaraFunctions.debug_f1_f2(f1, f2)
    f1.data .+= f2.data .* α
    return nothing
end


# For a MeshFunction with bosonic frequency, fermionic frequency, and momentum dependence,
# `meshes` returns union of three types.
# Then, it seems that this makes mesh_index type unstable, and getindex becomes type unstable.
# Here we specialized functions for each dimension to avoid this issue.
# Same for setindex.

function Base.:getindex(
    f :: MeshFunction{3, Q},
    x :: Vararg{Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon}, 3}
    ) :: Q where {Q <: Number}

    i1 = MatsubaraFunctions.mesh_index(x[1], meshes(f, 1))
    i2 = MatsubaraFunctions.mesh_index(x[2], meshes(f, 2))
    i3 = MatsubaraFunctions.mesh_index(x[3], meshes(f, 3))
    return f.data[i1, i2, i3]
end


function Base.:getindex(
    f :: MeshFunction{4, Q},
    x :: Vararg{Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon}, 4}
    ) :: Q where {Q <: Number}

    i1 = MatsubaraFunctions.mesh_index(x[1], meshes(f, 1))
    i2 = MatsubaraFunctions.mesh_index(x[2], meshes(f, 2))
    i3 = MatsubaraFunctions.mesh_index(x[3], meshes(f, 3))
    i4 = MatsubaraFunctions.mesh_index(x[4], meshes(f, 4))
    return f.data[i1, i2, i3, i4]
end

function Base.:getindex(
    f :: MeshFunction{5, Q},
    x :: Vararg{Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon}, 5}
    ) :: Q where {Q <: Number}

    i1 = MatsubaraFunctions.mesh_index(x[1], meshes(f, 1))
    i2 = MatsubaraFunctions.mesh_index(x[2], meshes(f, 2))
    i3 = MatsubaraFunctions.mesh_index(x[3], meshes(f, 3))
    i4 = MatsubaraFunctions.mesh_index(x[4], meshes(f, 4))
    i5 = MatsubaraFunctions.mesh_index(x[5], meshes(f, 5))
    return f.data[i1, i2, i3, i4, i5]
end


# To solve dispatch ambiguity
function Base.:getindex(f :: MeshFunction{3, Q}, x :: Vararg{Union{Int, UnitRange, Colon}, 3}
    ) where {Q <: Number}
    return f.data[x...]
end
function Base.:getindex(f :: MeshFunction{4, Q}, x :: Vararg{Union{Int, UnitRange, Colon}, 4}
    ) where {Q <: Number}
    return f.data[x...]
end
function Base.:getindex(f :: MeshFunction{5, Q}, x :: Vararg{Union{Int, UnitRange, Colon}, 5}
    ) where {Q <: Number}
    return f.data[x...]
end

function Base.:setindex!(
    f :: MeshFunction{3, Q},
    val :: Qp,
    x :: Vararg{Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon}, 3}
    ) where {Q <: Number, Qp <: Number}

    i1 = MatsubaraFunctions.mesh_index(x[1], meshes(f, 1))
    i2 = MatsubaraFunctions.mesh_index(x[2], meshes(f, 2))
    i3 = MatsubaraFunctions.mesh_index(x[3], meshes(f, 3))

    f.data[i1, i2, i3] = val
    return nothing
end

function Base.:setindex!(
    f :: MeshFunction{4, Q},
    val :: Qp,
    x :: Vararg{Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon}, 4}
    ) where {Q <: Number, Qp <: Number}

    i1 = MatsubaraFunctions.mesh_index(x[1], meshes(f, 1))
    i2 = MatsubaraFunctions.mesh_index(x[2], meshes(f, 2))
    i3 = MatsubaraFunctions.mesh_index(x[3], meshes(f, 3))
    i4 = MatsubaraFunctions.mesh_index(x[4], meshes(f, 4))

    f.data[i1, i2, i3, i4] = val
    return nothing
end


function Base.:setindex!(
    f :: MeshFunction{5, Q},
    val :: Qp,
    x :: Vararg{Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon}, 5}
    ) where {Q <: Number, Qp <: Number}

    i1 = MatsubaraFunctions.mesh_index(x[1], meshes(f, 1))
    i2 = MatsubaraFunctions.mesh_index(x[2], meshes(f, 2))
    i3 = MatsubaraFunctions.mesh_index(x[3], meshes(f, 3))
    i4 = MatsubaraFunctions.mesh_index(x[4], meshes(f, 4))
    i5 = MatsubaraFunctions.mesh_index(x[5], meshes(f, 5))

    f.data[i1, i2, i3, i4, i5] = val
    return nothing
end

# Avoid dispatch ambiguity
function Base.:setindex!(f :: MeshFunction{3, Q}, val :: Qp, x :: Vararg{Union{Int, UnitRange, Colon}, 3}
    ) where {Q <: Number, Qp <: Number}
    f.data[x...] = val
    return nothing
end

function Base.:setindex!(f :: MeshFunction{4, Q}, val :: Qp, x :: Vararg{Union{Int, UnitRange, Colon}, 4}
    ) where {Q <: Number, Qp <: Number}
    f.data[x...] = val
    return nothing
end

function Base.:setindex!(f :: MeshFunction{5, Q}, val :: Qp, x :: Vararg{Union{Int, UnitRange, Colon}, 5}
    ) where {Q <: Number, Qp <: Number}
    f.data[x...] = val
    return nothing
end


# Same for view

function Base.:view(f :: MeshFunction{4, Q}, x :: Vararg{Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon}, 4}
    ) where {Q <: Number}

    i1 = MatsubaraFunctions.mesh_index(x[1], meshes(f, 1))
    i2 = MatsubaraFunctions.mesh_index(x[2], meshes(f, 2))
    i3 = MatsubaraFunctions.mesh_index(x[3], meshes(f, 3))
    i4 = MatsubaraFunctions.mesh_index(x[4], meshes(f, 4))
    return view(f.data, i1, i2, i3, i4)
end

function Base.:view(f :: MeshFunction{5, Q}, x :: Vararg{Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon}, 5}
    ) where {Q <: Number}

    i1 = MatsubaraFunctions.mesh_index(x[1], meshes(f, 1))
    i2 = MatsubaraFunctions.mesh_index(x[2], meshes(f, 2))
    i3 = MatsubaraFunctions.mesh_index(x[3], meshes(f, 3))
    i4 = MatsubaraFunctions.mesh_index(x[4], meshes(f, 4))
    i5 = MatsubaraFunctions.mesh_index(x[5], meshes(f, 5))
    return view(f.data, i1, i2, i3, i4, i5)
end

# Avoid amgibuity

function Base.:view(f :: MeshFunction{4, Q}, x :: Vararg{Union{Int, UnitRange, Colon}, 4}
    ) where {Q <: Number}
    return view(f.data, x...)
end

function Base.:view(f :: MeshFunction{5, Q}, x :: Vararg{Union{Int, UnitRange, Colon}, 5}
    ) where {Q <: Number}
    return view(f.data, x...)
end


# euclidean in MatsubaraFunctions allocates due to type instability of basis(bz)
# Here I add type annotation to avoid this issue.
function MatsubaraFunctions.euclidean(
    k :: BrillouinPoint{2},
    mesh :: Mesh{MeshPoint{BrillouinPoint{2}}, BrillouinDomain{2}}
    ) :: SVector{2, Float64}

    b = basis(bz(mesh)) :: SMatrix{2, 2, Float64, 4}
    return b * (value(k) ./ bz(mesh).L)
end
