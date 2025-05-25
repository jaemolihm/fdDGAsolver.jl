struct RefVertex{Q}
    U    :: Q
    Fp_p :: MF_K3{Q}
    Fp_x :: MF_K3{Q}
    Ft_p :: MF_K3{Q}
    Ft_x :: MF_K3{Q}

    function RefVertex(
        U    :: Number,
        Fp_p :: MF_K3{Q},
        Fp_x :: MF_K3{Q},
        Ft_p :: MF_K3{Q},
        Ft_x :: MF_K3{Q},
        )    :: RefVertex{Q} where {Q}

        return new{Q}(Q(U), Fp_p, Fp_x, Ft_p, Ft_x)
    end

    function RefVertex(
        T :: Float64,
        U :: Float64,
          :: Type{Q} = ComplexF64,
        ) where {Q}

        # Null vertices
        gΩ = MatsubaraMesh(T, 1, Boson)
        gν = MatsubaraMesh(T, 1, Fermion)
        Fp = MeshFunction(gΩ, gν, gν; data_t = Q)
        Fx = MeshFunction(gΩ, gν, gν; data_t = Q)
        set!(Fp, 0)
        set!(Fx, 0)

        return new{Q}(Q(U), Fp, Fx, copy(Fp), copy(Fx)) :: RefVertex{Q}
    end
end

Base.eltype(::Type{<: RefVertex{Q}}) where {Q} = Q

function Base.show(io::IO, Γ::RefVertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(Γ.U), T = $(temperature(Γ)), K3 : $(numK3(Γ))")
end

function numK3(
    Λ :: RefVertex
    ) :: NTuple{2, Int64}

    return N(meshes(Λ.Fp_p, Val(1))), N(meshes(Λ.Fp_p, Val(2)))
end

function MatsubaraFunctions.set!(
    F :: RefVertex,
    val :: Number,
    ) :: Nothing

    set!(F.Fp_p, val)
    set!(F.Fp_x, val)
    set!(F.Ft_p, val)
    set!(F.Ft_x, val)

    return nothing
end

# getter methods
function MatsubaraFunctions.temperature(
    F :: RefVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(meshes(F.Fp_p, Val(1)))
end

# copy
function Base.:copy(
    F :: RefVertex{Q}
    ) :: RefVertex{Q} where {Q}

    return RefVertex(F.U, copy(F.Fp_p), copy(F.Fp_x), copy(F.Ft_p), copy(F.Ft_x))

end

# comparison
function Base.:(==)(
    F1 :: RefVertex{Q},
    F2 :: RefVertex{Q},
    )  :: Bool where {Q}
    return (F1.U == F2.U) && (F1.Fp_p == F2.Fp_p) && (F1.Fp_x == F2.Fp_x) && (F1.Ft_p == F2.Ft_p) && (F1.Ft_x == F2.Ft_x)
end


# evaluators for parallel spin component
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{pCh},
       :: Type{pSp},
    ; kwargs...
    )  :: Q where {Q}

    return F.Fp_p(Ω, ν, νp) + F.U
end

@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{tCh},
       :: Type{pSp},
    ; kwargs...
    )  :: Q where {Q}

    return F.Ft_p(Ω, ν, νp) + F.U
end

@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{aCh},
       :: Type{pSp},
    ; kwargs...
    )  :: Q where {Q}

    return -F.Ft_x(Ω, νp, ν) + F.U
end

# evaluators for crossed spin component
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{pCh},
       :: Type{xSp},
    ; kwargs...
    )  :: Q where {Q}

    return F.Fp_x(Ω, ν, νp) - F.U
end

@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{tCh},
       :: Type{xSp},
    ; kwargs...
    )  :: Q where {Q}

    return F.Ft_x(Ω, ν, νp) - F.U
end

@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{aCh},
       :: Type{xSp},
    ; kwargs...
    )  :: Q where {Q}

    return -F.Ft_p(Ω, νp, ν) - F.U
end

# evaluators for density component
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{Ch},
       :: Type{dSp},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag}

    return 2 * F(Ω, ν, νp, Ch, pSp) + F(Ω, ν, νp, Ch, xSp)
end

@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{Ch},
       :: Type{Sp},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    return bare_vertex(F, Ch, Sp)
end

@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency,
       :: Type{Ch},
       :: Type{Sp},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    return bare_vertex(F, Ch, Sp)
end

@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency,
       :: Type{Ch},
       :: Type{Sp},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    return bare_vertex(F, Ch, Sp)
end

@inline bare_vertex(F :: RefVertex) =  F.U
@inline bare_vertex(F :: RefVertex, :: Type{pSp}) =  F.U
@inline bare_vertex(F :: RefVertex, :: Type{xSp}) = -F.U
@inline bare_vertex(F :: RefVertex, :: Type{dSp}) =  F.U
@inline bare_vertex(F :: RefVertex, :: Type{Ch}, :: Type{Sp}) where {Ch <: ChannelTag, Sp <: SpinTag} = bare_vertex(F, Sp)   # TODO: remove

# save to HDF5
function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    F     :: RefVertex
    )     :: Nothing

    grp = create_group(file, label)

    attributes(grp)["U"] = F.U

    MatsubaraFunctions.save!(file, label * "/Fp_p", F.Fp_p)
    MatsubaraFunctions.save!(file, label * "/Fp_x", F.Fp_x)
    MatsubaraFunctions.save!(file, label * "/Ft_p", F.Ft_p)
    MatsubaraFunctions.save!(file, label * "/Ft_x", F.Ft_x)

    return nothing
end

function load_refvertex(
    file  :: HDF5.File,
    label :: String
    )     :: RefVertex

    U = read_attribute(file[label], "U")

    Fp_p = load_mesh_function(file, label * "/Fp_p")
    Fp_x = load_mesh_function(file, label * "/Fp_x")
    Ft_p = load_mesh_function(file, label * "/Ft_p")
    Ft_x = load_mesh_function(file, label * "/Ft_x")

    return RefVertex(U, Fp_p, Fp_x, Ft_p, Ft_x)
end
