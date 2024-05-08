abstract type AbstractVertex{Q}; end
Base.eltype(::Type{<: AbstractVertex{Q}}) where {Q} = Q

# Force implementation of channel_type for AbstractVertex
channel_type(::Type{<: AbstractVertex}) = error("Not implemented")

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

channel_type(::Type{Vertex}) = Channel

Base.eltype(::Type{<: Vertex{Q}}) where {Q} = Q

function Base.show(io::IO, Γ::AbstractVertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(bare_vertex(Γ)), T = $(temperature(Γ))\n")
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

function MatsubaraFunctions.mult_add!(
    F1 :: AbstractVertex,
    F2 :: AbstractVertex,
    val :: Number
    )  :: Nothing

    mult_add!(F1.γp, F2.γp, val)
    mult_add!(F1.γt, F2.γt, val)
    mult_add!(F1.γa, F2.γa, val)

    return nothing
end

# maximum absolute value
function MatsubaraFunctions.absmax(
    F :: AbstractVertex
    ) :: Float64

    return max(absmax(F.γp), absmax(F.γt), absmax(F.γa))
end

# comparison
function Base.:(==)(
    F1 :: AbstractVertex{Q},
    F2 :: AbstractVertex{Q},
    )  :: Bool where {Q}
    return (F1.F0 == F2.F0) && (F1.γa == F2.γa) && (F1.γp == F2.γp) && (F1.γt == F2.γt)
end

function Base.length(
    F :: AbstractVertex,
    ) :: Int
    return length(F.γp) + length(F.γt) + length(F.γa)
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
        val += F.F0(Ω, ν, νp, Ch, pSp)
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
        val += F.F0(Ω, ν, νp, Ch, pSp)
    end

    # This function is called only if either ν or νp is an InfiniteMatsubaraFrequency.
    # Otherwise, the specific case of having all MatsubaraFrequency's is called.

    # If ν or νp is an InfiniteMatsubaraFrequency, reducible vertices is nonzero
    # only for the same channel evaluated.

    if Ch === pCh && γp
        val += F.γp(Ω, ν, νp)
    end

    if Ch === tCh && γt
        val += F.γt(Ω, ν, νp)
    end

    if Ch === aCh && γa
        val += F.γa(Ω, ν, νp)
    end

    return val
end


# evaluators for crossed spin component
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    if Ch === pCh
        return -F(Ω, ν, Ω - νp, pCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)

    elseif Ch === tCh
        return -F(Ω, ν, νp, aCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)

    elseif Ch === aCh
        return -F(Ω, ν, νp, tCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)

    else
        throw(ArgumentError("Invalid channel $Ch"))
    end
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


@inline bare_vertex(F :: AbstractVertex) =  bare_vertex(F.F0)
@inline bare_vertex(F :: AbstractVertex, :: Type{Sp}) where {Sp <: SpinTag} = bare_vertex(F.F0, Sp)
@inline bare_vertex(F :: AbstractVertex, :: Type{Ch}, :: Type{Sp}) where {Ch <: ChannelTag, Sp <: SpinTag} = bare_vertex(F, Sp)  # TODO: remove

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
    ) :: Nothing

    reduce!(F.γp)
    reduce!(F.γt)
    reduce!(F.γa)

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
          :: Type{T},
    file  :: HDF5.File,
    label :: String
    )     :: T where {T <: AbstractVertex}

    if haskey(attributes(file[label * "/F0"]), "U")
        F0 = load_refvertex(file, label * "/F0")
    else
        try
            F0 = load_vertex(Vertex, file, label * "/F0")
        catch
            try
                F0 = load_vertex(NL_Vertex, file, label * "/F0")
            catch
                try
                    F0 = load_vertex(NL2_Vertex, file, label * "/F0")
                catch
                    try
                        F0 = load_vertex(NL3_Vertex, file, label * "/F0")
                    catch
                        F0 = load_vertex(NL2_MBEVertex, file, label * "/F0")
                    end
                end
            end
        end
    end
    γp = load_channel(channel_type(T), file, label * "/γp")
    γt = load_channel(channel_type(T), file, label * "/γt")
    γa = load_channel(channel_type(T), file, label * "/γa")

    return T(F0, γp, γt, γa)
end
