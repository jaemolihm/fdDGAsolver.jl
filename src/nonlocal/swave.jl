# S-wave Brillouin point. When used to evaluate vertex, integrate over the corresponding momentum.
#
# When using getindex with an SWaveBrillouinPoint the respective momentum argument is summed over.
# This is used in the BSE for the s-wave approximation of the vertex (i.e. where the vertex
# is approximated to be independent of the fermionic momentum).
#
# This file implements the basic implementation for MeshFunction types.
#
# The Channel and Vertex implements short-circuit evaluation for the SWaveBrillouinPoint,
# taking into account the channel conversion of the momentum.
# For example, suppose we want to evaluate the a-channel K1 vertex in the p-channel momentum `(P, k, kSW)`.
# A brute-force implementation would be `∑_{q} K1(convert_momentum(P, k, q, pCh, aCh)[1]) / N_q`.
# But, one can do this more efficiently via `∑_{q} K1(k - q) / N_q = K1(kSW)`, which indices
# the MeshFunction object only a single time, instead of `N_q` times.


struct SWaveBrillouinPoint <: AbstractValue; end
const kSW = SWaveBrillouinPoint()

MatsubaraFunctions.fold_back(:: SWaveBrillouinPoint, :: KMesh) = SWaveBrillouinPoint()
MatsubaraFunctions.mesh_index_bc(:: SWaveBrillouinPoint, :: KMesh) = SWaveBrillouinPoint()

Base.:+(::BrillouinPoint, ::SWaveBrillouinPoint) = SWaveBrillouinPoint()
Base.:-(::BrillouinPoint, ::SWaveBrillouinPoint) = SWaveBrillouinPoint()
Base.:+(::SWaveBrillouinPoint, ::BrillouinPoint) = SWaveBrillouinPoint()
Base.:-(::SWaveBrillouinPoint, ::BrillouinPoint) = SWaveBrillouinPoint()
Base.:+(::SWaveBrillouinPoint, ::SWaveBrillouinPoint) = SWaveBrillouinPoint()
Base.:-(::SWaveBrillouinPoint, ::SWaveBrillouinPoint) = SWaveBrillouinPoint()



function Base.getindex(
    f  :: Union{NL_MF_K1{Q}, NL_MF_G{Q}},
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
    P  :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i3 = MatsubaraFunctions.mesh_index(P,  meshes(f, Val(3)))

    return sum(view(f, i1, i2, i3, :)) / length(meshes(f, Val(4)))

end


function Base.getindex(
    f  :: NL2_MF_K2{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
    k1 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i3 = MatsubaraFunctions.mesh_index(k1, meshes(f, Val(3)))

    return sum(view(f, i1, i2, i3, :)) / length(meshes(f, Val(4)))

end

function Base.getindex(
    f  :: NL2_MF_K2{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    k2 :: Union{MeshPoint, <: AbstractValue, Int},
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
