struct Vertex{Q, VT}
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

function Base.show(io::IO, Γ::Vertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(Γ.F0.U), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

# getter methods
function MatsubaraFunctions.temperature(
    F :: Vertex
    ) :: Float64

    return MatsubaraFunctions.temperature(F.γp)
end

function numK1(
    F :: Vertex
    ) :: Int64

    return numK1(F.γp)
end

function numK2(
    F :: Vertex
    ) :: NTuple{2, Int64}

    return numK2(F.γp)
end

function numK3(
    F :: Vertex
    ) :: NTuple{2, Int64}

    return numK3(F.γp)
end

# setter methods
function MatsubaraFunctions.set!(
    F1 :: Vertex,
    F2 :: Vertex
    )  :: Nothing

    set!(F1.γp, F2.γp)
    set!(F1.γt, F2.γt)
    set!(F1.γa, F2.γa)

    return nothing
end

function reset!(
    F :: Vertex
    ) :: Nothing

    reset!(F.γp)
    reset!(F.γt)
    reset!(F.γa)

    return nothing
end

# copy
function Base.:copy(
    F :: Vertex{Q}
    ) :: Vertex{Q} where {Q}

    return Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

# addition
function MatsubaraFunctions.add!(
    F1 :: Vertex,
    F2 :: Vertex
    )  :: Nothing

    add!(F1.γp, F2.γp)
    add!(F1.γt, F2.γt)
    add!(F1.γa, F2.γa)

    return nothing
end

# maximum absolute value
function MatsubaraFunctions.absmax(
    F :: Vertex
    ) :: Float64

    return max(absmax(F.γp), absmax(F.γt), absmax(F.γa))
end

# flatten into vector
function MatsubaraFunctions.flatten!(
    F :: Vertex,
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
    F :: Vertex{Q}
    ) :: Vector{Q} where {Q}

    xp = flatten(F.γp)
    xt = flatten(F.γt)
    xa = flatten(F.γa)

    return vcat(xp, xt, xa)
end

# unflatten from vector
function MatsubaraFunctions.unflatten!(
    F :: Vertex,
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

# evaluators for parallel spin component
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{pCh},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, pCh, pSp)
    end

    if γp
        val += F.γp(Ω, ν, νp)
    end

    if γt
        val += F.γt(Ω - ν - νp, νp, ν)
    end

    if γa
        val += F.γa(ν - νp, Ω - ν, νp)
    end

    return val
end

@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{tCh},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, tCh, pSp)
    end

    if γp
        val += F.γp(Ω + ν + νp, νp, ν)
    end

    if γt
        val += F.γt(Ω, ν, νp)
    end

    if γa
        val += F.γa(νp - ν, Ω + ν, ν)
    end

    return val
end

@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{aCh},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, aCh, pSp)
    end

    if γp
        val += F.γp(Ω + νp + ν, Ω + νp, νp)
    end

    if γt
        val += F.γt(ν - νp, νp, Ω + νp)
    end

    if γa
        val += F.γa(Ω, ν, νp)
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

@inline bare_vertex(F :: Vertex, :: Type{Ch}, :: Type{Sp}) where {Ch <: ChannelTag, Sp <: SpinTag} = bare_vertex(F.F0, Ch, Sp)
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
    F :: Vertex
    ) :: Nothing

    reduce!(F.γp)
    reduce!(F.γt)
    reduce!(F.γa)

    return nothing
end

# save to HDF5
function save_vertex!(
    file  :: HDF5.File,
    label :: String,
    F     :: Vertex
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