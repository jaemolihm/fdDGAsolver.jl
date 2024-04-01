using MatsubaraFunctions: MeshFunction

abstract type AbstractVertex{DD, Q <: Number} end
channel_freq(v::AbstractVertex) = v.channel_freq

"""
    Vertex_K1{1, Q <: Number, MT <: NTuple{1, Mesh}, AT <: AbstractArray{Q, DD}, MF <: MeshFunction{1, Q, MT, AT}}

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


function Base.show(io::IO, v::Vertex_K1)
    print(io, "$(nameof(typeof(v))), channel_freq = $(v.channel_freq)\n")
    print(io, "meshes:\n")
    for (i, mesh) in enumerate(v.data.meshes)
        print(io, "    $i. $(typeof(value(first(mesh.points)))), N = $(mesh.domain[:N])")
    end
end

Base.getindex(f :: AbstractVertex{DD}, x :: Vararg{Any, DD}) where {DD} = getindex(f.data, x...)

function (f :: AbstractVertex)(v1, v2, w, channel_freq :: Symbol = f.channel_freq)
    if channel_freq === f.channel_freq
        f.data(w; lim = f.lim)
    else
        v1_, v2_, w_ = convert_channel(f.channel_freq, channel_freq, v1, v2, w)
        f.data(w_; lim = f.lim)
    end
end

export
    Vertex_K1