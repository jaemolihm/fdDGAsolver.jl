# S-wave Brillouin point. When used to evaluate vertex, integrate over the corresponding momentum.
struct SWaveBrillouinPoint <: AbstractValue; end
const kSW = SWaveBrillouinPoint()

MatsubaraFunctions.fold_back(:: SWaveBrillouinPoint, :: KMesh) = SWaveBrillouinPoint()


function Base.getindex(
    f  :: NL_MF_K1{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))

    return sum(view(f, i1, :)) / length(meshes(f, Val(2)))

end

function Base.getindex(
    f  :: NL_MF_K2{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))

    return sum(view(f, i1, i2, :)) / length(meshes(f, Val(3)))

end

function Base.getindex(
    f  :: NL_MF_K3{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
    w3 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i3 = MatsubaraFunctions.mesh_index(w3, meshes(f, Val(3)))

    return sum(view(f, i1, i2, i3, :)) / length(meshes(f, Val(4)))

end

function Base.getindex(
    f  :: NL_MF_Π{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
    P  :: BrillouinPoint,
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i3 = MatsubaraFunctions.mesh_index(P,  meshes(f, Val(3)))

    return sum(view(f, i1, i2, i3, :)) / length(meshes(f, Val(4)))

end


# NL2_MF_K2 and NL_MF_Π are the same
# function Base.getindex(
#     f  :: NL2_MF_K2{Q},
#     w1 :: Union{MeshPoint, <: AbstractValue, Int},
#     w2 :: Union{MeshPoint, <: AbstractValue, Int},
#     k1 :: BrillouinPoint,
#        :: SWaveBrillouinPoint,
#     )  :: Q where {Q}

#     i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, 1))
#     i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, 2))
#     i3 = MatsubaraFunctions.mesh_index(k1, meshes(f, 3))

#     return sum(view(f, i1, i2, i3, :)) / length(meshes(f, 4))

# end

function Base.getindex(
    f  :: NL2_MF_K2{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    k2 :: BrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i4 = MatsubaraFunctions.mesh_index(k2, meshes(f, Val(4)))

    return sum(view(f, i1, i2, :, i4)) / length(meshes(f, Val(3)))

end

function Base.getindex(
    f  :: NL2_MF_K2{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))

    return sum(view(f, i1, i2, :, :)) / length(meshes(f, Val(3))) / length(meshes(f, Val(4)))

end
