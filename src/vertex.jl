using MatsubaraFunctions: MeshFunction

abstract type AbstractVertex{DD, Q <: Number} end
channel_freq(v::AbstractVertex) = v.channel_freq

"""
    Vertex_K1{DD, Q <: Number, MT <: NTuple{1, Mesh}, AT <: AbstractArray{Q, DD}, MF <: MeshFunction{1, Q, MT, AT}}

Vertex_K1 type with fields:
* `data :: MF` : `MeshFunction` storing the data
* `channel_freq :: Symbol` : Frequency channel. `:a`, `:p`, or `t`
* `lim :: Q` : Extrapolated constant value outside the mesh.
"""
struct Vertex_K1{DD, Q <: Number, MT <: NTuple{DD, Mesh}, AT <: AbstractArray{Q, DD}, MF <: MeshFunction{DD, Q, MT, AT}} <: AbstractVertex{DD, Q}
    data :: MF
    channel_freq :: Symbol
    lim :: Q
end

struct Vertex_K2{DD, Q <: Number, MT <: NTuple{DD, Mesh}, AT <: AbstractArray{Q, DD}, MF <: MeshFunction{DD, Q, MT, AT}} <: AbstractVertex{DD, Q}
    data :: MF
    channel_freq :: Symbol
    lim :: Q
end

struct Vertex_K2p{DD, Q <: Number, MT <: NTuple{DD, Mesh}, AT <: AbstractArray{Q, DD}, MF <: MeshFunction{DD, Q, MT, AT}} <: AbstractVertex{DD, Q}
    data :: MF
    channel_freq :: Symbol
    lim :: Q
end

struct Vertex_K3{DD, Q <: Number, MT <: NTuple{DD, Mesh}, AT <: AbstractArray{Q, DD}, MF <: MeshFunction{DD, Q, MT, AT}} <: AbstractVertex{DD, Q}
    data :: MF
    channel_freq :: Symbol
    lim :: Q
end


function Base.show(io::IO, v::AbstractVertex)
    print(io, "$(nameof(typeof(v))), channel_freq = $(v.channel_freq)\n")
    print(io, "meshes:")
    for (i, mesh) in enumerate(v.data.meshes)
        print(io, "\n    $i. $(typeof(value(first(mesh.points)))), N = $(domain(mesh).N)")
    end
end

Base.size(v :: AbstractVertex) = length.(meshes(v.data))

Base.getindex(f :: AbstractVertex{DD}, x :: Vararg{Any, DD}) where {DD} = getindex(f.data, x...)

function (f :: AbstractVertex)(v1, v2, w, channel_freq :: Symbol)
    v1_, v2_, w_ = convert_channel(f.channel_freq, channel_freq, v1, v2, w)
    f(v1_, v2_, w_)
end

(f :: Vertex_K1)(v1, v2, w) = f.data(w; lim = f.lim)
(f :: Vertex_K2)(v1, v2, w) = f.data(v1, w; lim = f.lim)
(f :: Vertex_K2p)(v1, v2, w) = f.data(v2, w; lim = f.lim)
(f :: Vertex_K3)(v1, v2, w) = f.data(v1, v2, w; lim = f.lim)

export
    Vertex_K1,
    Vertex_K2,
    Vertex_K2p,
    Vertex_K3