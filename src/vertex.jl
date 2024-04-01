using MatsubaraFunctions: MeshFunction

abstract type AbstractSU2Vertex{DD, Q <: Number} end
channel_freq(v::AbstractSU2Vertex) = v.channel_freq
channel_spin(v::AbstractSU2Vertex) = v.channel_spin

"""
    SU2Vertex_K1{1, Q <: Number, MT <: NTuple{1, Mesh}, AT <: AbstractArray{Q, DD}, MF <: MeshFunction{1, Q, MT, AT}}

SU2Vertex_K1 type with fields:
* `data :: MF` : `MeshFunction` storing the data
* `channel_freq :: Symbol` : Frequency channel. `:a`, `:p`, or `t`
* `channel_spin :: Symbol` : Spin channel. `:Da`, `:Ma`, `:Sp`, `:Tp`, `:Dt`, or `:Mt`
"""
struct SU2Vertex_K1{DD, Q <: Number, MT <: NTuple{DD, Mesh}, AT <: AbstractArray{Q, DD}, MF <: MeshFunction{DD, Q, MT, AT}} <: AbstractSU2Vertex{DD, Q}
    data :: MF
    channel_freq :: Symbol
    channel_spin :: Symbol
    lim :: Q
end


function Base.show(io::IO, v::SU2Vertex_K1)
    print(io, "$(nameof(typeof(v))), channel_freq = $(v.channel_freq), channel_spin = $(v.channel_spin)\n")
    print(io, "meshes:\n")
    for (i, mesh) in enumerate(v.data.meshes)
        print(io, "    $i. $(typeof(value(first(mesh.points)))), N = $(mesh.domain[:N])")
    end
end

Base.getindex(f :: AbstractSU2Vertex{DD}, x :: Vararg{Any, DD}) where {DD} = getindex(f.data, x...)

function (f :: AbstractSU2Vertex)(v1, v2, w, channel_freq :: Symbol = f.channel_freq)
    if channel_freq === f.channel_freq
        f.data(w; lim = f.lim)
    else
        v1_, v2_, w_ = convert_channel(f.channel_freq, channel_freq, v1, v2, w)
        f.data(w_; lim = f.lim)
    end
end

export
    SU2Vertex_K1