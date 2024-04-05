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


# To solve dispatch ambiguity
function Base.:getindex(f :: MeshFunction{3, Q}, x :: Vararg{Union{Int, UnitRange, Colon}, 3}
    ) where {Q <: Number}
    return f.data[x...]
end
function Base.:getindex(f :: MeshFunction{4, Q}, x :: Vararg{Union{Int, UnitRange, Colon}, 4}
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
