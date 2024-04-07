abstract type AbstractVertex{Q}; end
Base.eltype(::Type{<: AbstractVertex{Q}}) where {Q} = Q

struct Vertex{Q, VT} <: AbstractVertex{Q}
    F0 :: VT
    γp :: Channel{Q}
    γt :: Channel{Q}
    γa :: Channel{Q}

    function Vertex(
        F0 :: VT,
        γp :: Channel{Q},
        γt :: Channel{Q},
        γa :: Channel{Q},
        )  :: Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    function Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        ) where {VT}

        Q = eltype(F0)

        γ = Channel(T, numK1, numK2, numK3, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: Vertex{Q}
    end
end

Base.eltype(::Type{<: Vertex{Q}}) where {Q} = Q

function Base.show(io::IO, Γ::Vertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(Γ.F0.U), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

# getter methods
function MatsubaraFunctions.temperature(
    F :: AbstractVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(F.γp)
end

function numK1(
    F :: AbstractVertex
    ) :: Int64

    return numK1(F.γp)
end

function numK2(
    F :: AbstractVertex
    ) :: NTuple{2, Int64}

    return numK2(F.γp)
end

function numK3(
    F :: AbstractVertex
    ) :: NTuple{2, Int64}

    return numK3(F.γp)
end

# setter methods
function MatsubaraFunctions.set!(
    F1 :: AbstractVertex,
    F2 :: AbstractVertex
    )  :: Nothing

    set!(F1.γp, F2.γp)
    set!(F1.γt, F2.γt)
    set!(F1.γa, F2.γa)

    return nothing
end

function MatsubaraFunctions.set!(
    F :: AbstractVertex,
    val :: Number,
    ) :: Nothing

    set!(F.γp, val)
    set!(F.γt, val)
    set!(F.γa, val)

    return nothing
end

# addition
function MatsubaraFunctions.add!(
    F1 :: AbstractVertex,
    F2 :: AbstractVertex
    )  :: Nothing

    add!(F1.γp, F2.γp)
    add!(F1.γt, F2.γt)
    add!(F1.γa, F2.γa)

    return nothing
end

# maximum absolute value
function MatsubaraFunctions.absmax(
    F :: AbstractVertex
    ) :: Float64

    return max(absmax(F.γp), absmax(F.γt), absmax(F.γa))
end

# flatten into vector
function MatsubaraFunctions.flatten!(
    F :: AbstractVertex,
    x :: AbstractVector
    ) :: Nothing

    offset = 0
    len_γ  = length(F.γp)

    flatten!(F.γp, @view x[1 + offset : offset + len_γ]); offset += len_γ
    flatten!(F.γt, @view x[1 + offset : offset + len_γ]); offset += len_γ
    flatten!(F.γa, @view x[1 + offset : offset + len_γ]); offset += len_γ

    @assert offset == length(x) "Dimension mismatch between vertex and target vector"
    return nothing
end

function MatsubaraFunctions.flatten(
    F :: AbstractVertex{Q}
    ) :: Vector{Q} where {Q}

    xp = flatten(F.γp)
    xt = flatten(F.γt)
    xa = flatten(F.γa)

    return vcat(xp, xt, xa)
end

# unflatten from vector
function MatsubaraFunctions.unflatten!(
    F :: AbstractVertex,
    x :: AbstractVector
    ) :: Nothing

    offset = 0
    len_γ  = length(F.γp)

    unflatten!(F.γp, @view x[1 + offset : offset + len_γ]); offset += len_γ
    unflatten!(F.γt, @view x[1 + offset : offset + len_γ]); offset += len_γ
    unflatten!(F.γa, @view x[1 + offset : offset + len_γ]); offset += len_γ

    @assert offset == length(x) "Dimension mismatch between vertex and target vector"
    return nothing
end


# copy
function Base.:copy(
    F :: Vertex{Q}
    ) :: Vertex{Q} where {Q}

    return Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end


# evaluators for parallel spin component
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{Ch},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, Ch, pSp; F0, γp, γt, γa)
    end

    if γp
        val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)...)
    end

    if γt
        val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)...)
    end

    if γa
        val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)...)
    end

    return val
end



# Special cases where either ν or νp is an InfiniteMatsubaraFrequency
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, Ch, pSp; F0, γp, γt, γa)
    end

    # This function is called only if either ν or νp is an InfiniteMatsubaraFrequency.
    # Otherwise, the specific case of having all MatsubaraFrequency's is called.

    # If ν or νp is an InfiniteMatsubaraFrequency, reducible vertices is nonzero
    # only for the same channel evaluated.

    if Ch === pCh && γp
        val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)...)
    end

    if Ch === tCh && γt
        val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)...)
    end

    if Ch === aCh && γa
        val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)...)
    end

    return val
end


# evaluators for crossed spin component
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{pCh},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    return -F(Ω, ν, Ω - νp, pCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)
end

@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{tCh},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    return -F(Ω, νp, ν, aCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)
end

@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{aCh},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    return -F(Ω, νp, ν, tCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)
end

# evaluators for density spin component
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{dSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    val += F(Ω, ν, νp, Ch, pSp; F0, γp, γt, γa) * 2

    val += F(Ω, ν, νp, Ch, xSp; F0, γp, γt, γa)

    return val
end


@inline bare_vertex(F :: Vertex) =  bare_vertex(F.F0)
@inline bare_vertex(F :: Vertex, :: Type{Ch}, :: Type{Sp}) where {Ch <: ChannelTag, Sp <: SpinTag} = bare_vertex(F.F0, Ch, Sp)

# # build full vertex in given frequency convention and spin component
# function mk_vertex(
#     F   :: Vertex,
#     gΩ  :: MatsubaraGrid,
#     gν  :: MatsubaraGrid,
#     gνp :: MatsubaraGrid,
#         :: Type{CT},
#         :: Type{ST}
#     ;
#     F0  :: Bool = true,
#     γp  :: Bool = true,
#     γt  :: Bool = true,
#     γa  :: Bool = true
#     )   :: MatsubaraFunction{3, 1, 4, Float64} where {CT <: ChannelTag, ST <: SpinTag}

#     f = MatsubaraFunction((gΩ, gν, gνp), 1, Float64)

#     Threads.@threads for i in eachindex(f.data)
#         f[i] = F(first(to_Matsubara(f, i))..., CT, ST; F0 = F0, γp = γp, γt = γt, γa = γa)
#     end

#     return f
# end

# reducer
function reduce!(
    F :: AbstractVertex
    ;
    max_class :: Int = 3,
    ) :: Nothing

    reduce!(F.γp; max_class)
    reduce!(F.γt; max_class)
    reduce!(F.γa; max_class)

    return nothing
end

# save to HDF5
function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    F     :: AbstractVertex
    )     :: Nothing

    MatsubaraFunctions.save!(file, label * "/F0", F.F0)
    MatsubaraFunctions.save!(file, label * "/γp", F.γp)
    MatsubaraFunctions.save!(file, label * "/γt", F.γt)
    MatsubaraFunctions.save!(file, label * "/γa", F.γa)

    return nothing
end

# load from HDF5
function load_vertex(
    file  :: HDF5.File,
    label :: String
    )     :: Vertex

    F0 = load_refvertex(file, label * "/F0")
    γp = load_channel(file, label * "/γp")
    γt = load_channel(file, label * "/γt")
    γa = load_channel(file, label * "/γa")

    return Vertex(F0, γp, γt, γa)
end
