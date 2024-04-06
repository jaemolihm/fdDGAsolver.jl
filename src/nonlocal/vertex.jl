struct NL_Vertex{Q, VT}
    F0 :: VT
    γp :: NL_Channel{Q}
    γt :: NL_Channel{Q}
    γa :: NL_Channel{Q}

    function NL_Vertex(
        F0 :: VT,
        γp :: NL_Channel{Q},
        γt :: NL_Channel{Q},
        γa :: NL_Channel{Q},
        )  :: NL_Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    function NL_Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
        ) where {VT}

        Q = eltype(F0)

        γ = NL_Channel(T, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: NL_Vertex{Q}
    end
end

Base.eltype(::Type{<: NL_Vertex{Q}}) where {Q} = Q

function Base.show(io::IO, Γ::NL_Vertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(Γ.F0.U), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

# getter methods
function MatsubaraFunctions.temperature(
    F :: NL_Vertex
    ) :: Float64

    return MatsubaraFunctions.temperature(F.γp)
end

function get_P_mesh(
    F :: NL_Vertex
    ) :: KMesh

    return get_P_mesh(F.γp)
end

function numK1(
    F :: NL_Vertex
    ) :: Int64

    return numK1(F.γp)
end

function numK2(
    F :: NL_Vertex
    ) :: NTuple{2, Int64}

    return numK2(F.γp)
end

function numK3(
    F :: NL_Vertex
    ) :: NTuple{2, Int64}

    return numK3(F.γp)
end

function numP(
    F :: NL_Vertex
    ) :: Int64

    return numP(F.γp)
end

# setter methods
function MatsubaraFunctions.set!(
    F1 :: NL_Vertex,
    F2 :: NL_Vertex
    )  :: Nothing

    set!(F1.γp, F2.γp)
    set!(F1.γt, F2.γt)
    set!(F1.γa, F2.γa)

    return nothing
end

function MatsubaraFunctions.set!(
    F   :: NL_Vertex,
    val :: Number,
    )   :: Nothing

    set!(F.γp, val)
    set!(F.γt, val)
    set!(F.γa, val)

    return nothing
end

# copy
function Base.:copy(
    F :: NL_Vertex{Q}
    ) :: NL_Vertex{Q} where {Q}

    return NL_Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

# addition
function MatsubaraFunctions.add!(
    F1 :: NL_Vertex,
    F2 :: NL_Vertex
    )  :: Nothing

    add!(F1.γp, F2.γp)
    add!(F1.γt, F2.γt)
    add!(F1.γa, F2.γa)

    return nothing
end

# maximum absolute value
function MatsubaraFunctions.absmax(
    F :: NL_Vertex
    ) :: Float64

    return max(absmax(F.γp), absmax(F.γt), absmax(F.γa))
end

# flatten into vector
function MatsubaraFunctions.flatten!(
    F :: NL_Vertex,
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
    F :: NL_Vertex{Q}
    ) :: Vector{Q} where {Q}

    xp = flatten(F.γp)
    xt = flatten(F.γt)
    xa = flatten(F.γa)

    return vcat(xp, xt, xa)
end

# unflatten from vector
function MatsubaraFunctions.unflatten!(
    F :: NL_Vertex,
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
@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp; F0, γp, γt, γa)
    end

    if γp
        val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)...,
                    convert_momentum( P, k, kp, Ch, pCh)...)
    end

    if γt
        val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)...,
                    convert_momentum( P, k, kp, Ch, tCh)...)
    end

    if γa
        val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)...,
                    convert_momentum( P, k, kp, Ch, aCh)...)
    end

    return val
end



# Special cases where either ν or νp is an InfiniteMatsubaraFrequency
@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp; F0, γp, γt, γa)
    end

    # This function is called only if either ν or νp is an InfiniteMatsubaraFrequency.
    # Otherwise, the specific case of having all MatsubaraFrequency's is called.

    # If ν or νp is an InfiniteMatsubaraFrequency, reducible vertices is nonzero
    # only for the same channel evaluated.

    if Ch === pCh && γp
        val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)...,
                    convert_momentum( P, k, kp, Ch, pCh)...)
    end

    if Ch === tCh && γt
        val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)...,
                    convert_momentum( P, k, kp, Ch, tCh)...)
    end

    if Ch === aCh && γa
        val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)...,
                    convert_momentum( P, k, kp, Ch, aCh)...)
    end

    return val
end


# evaluators for crossed spin component
@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: BrillouinPoint,
       :: Type{pCh},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    return -F(Ω, ν, Ω - νp, P, k, P - kp, pCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)
end

@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{tCh},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    return -F(Ω, νp, ν, P, kp, k, aCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)
end

@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{aCh},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q}

    return -F(Ω, νp, ν, P, kp, k, tCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)
end

# evaluators for density spin component
@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Ch},
       :: Type{dSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    val += F(Ω, ν, νp, P, k, kp, Ch, pSp; F0, γp, γt, γa) * 2

    val += F(Ω, ν, νp, P, k, kp, Ch, xSp; F0, γp, γt, γa)

    return val
end


# S wave evaluation

@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    kp :: BrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp; F0, γp, γt, γa)
    end

    # k isa SWaveBrillouinPoint, sum over the k momentum.
    # We use the fact that NL_Vertex has only bosonic momentum dependence.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., P)
        else
            # Vertices in the different channel will be integrated over the bosonic momentum.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., SWaveBrillouinPoint())
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., P)
        else
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., SWaveBrillouinPoint())
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., P)
        else
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., SWaveBrillouinPoint())
        end
    end

    return val
end

@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: SWaveBrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp; F0, γp, γt, γa)
    end

    # k isa SWaveBrillouinPoint, sum over the k momentum.
    # We use the fact that NL_Vertex has only bosonic momentum dependence.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., P)
        else
            # Vertices in the different channel will be integrated over the bosonic momentum.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., SWaveBrillouinPoint())
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., P)
        else
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., SWaveBrillouinPoint())
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., P)
        else
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., SWaveBrillouinPoint())
        end
    end

    return val
end



@inline bare_vertex(F :: NL_Vertex) =  bare_vertex(F.F0)
@inline bare_vertex(F :: NL_Vertex, :: Type{Ch}, :: Type{Sp}) where {Ch <: ChannelTag, Sp <: SpinTag} = bare_vertex(F.F0, Ch, Sp)


# reducer
function reduce!(
    F :: NL_Vertex
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
    F     :: NL_Vertex
    )     :: Nothing

    MatsubaraFunctions.save!(file, label * "/F0", F.F0)
    MatsubaraFunctions.save!(file, label * "/γp", F.γp)
    MatsubaraFunctions.save!(file, label * "/γt", F.γt)
    MatsubaraFunctions.save!(file, label * "/γa", F.γa)

    return nothing
end

# load from HDF5
function load_nonlocal_vertex(
    file  :: HDF5.File,
    label :: String
    )     :: NL_Vertex

    F0 = load_refvertex(file, label * "/F0")
    γp = load_channel(file, label * "/γp")
    γt = load_channel(file, label * "/γt")
    γa = load_channel(file, label * "/γa")

    return NL_Vertex(F0, γp, γt, γa)
end
