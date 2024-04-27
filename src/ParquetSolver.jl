abstract type AbstractSolver{Q}; end

Base.eltype(::Type{<: AbstractSolver{Q}}) where {Q} = Q

mutable struct ParquetSolver{Q, VT, RefVT} <: AbstractSolver{Q}
    # Bare Green function
    Gbare :: MF_G{Q}

    # single-particle Green's function and bubbles for reference system
    G0::MF_G{Q}
    Π0pp::MF_Π{Q}
    Π0ph::MF_Π{Q}

    # self-energy for reference system
    Σ0::MF_G{Q}

    # two-particle vertex for reference system
    F0::RefVT

    # single-particle Green's function and bubbles for target system
    G::MF_G{Q}
    Πpp::MF_Π{Q}
    Πph::MF_Π{Q}

    # self-energy for target system
    Σ::MF_G{Q}

    # channel-decomposed two-particle vertex for target system
    F::VT

    # channel-decomposed two-particle vertex buffer
    Fbuff::Vertex{Q, RefVertex{Q}}

    # channel-decomposed two-particle vertex buffer for left part
    FL::Vertex{Q, RefVertex{Q}}

    # K2 vertices used in the SDE
    Lpp  :: MF_K2{Q}
    Lph  :: MF_K2{Q}
    L0pp :: MF_K2{Q}
    L0ph :: MF_K2{Q}

    # symmetry group for the self energy
    SGΣ::SymmetryGroup

    # symmetry groups for the particle-particle channel of F
    SGpp::Vector{SymmetryGroup}

    # symmetry groups for the particle-hole channels of F
    SGph::Vector{SymmetryGroup}

    # symmetry groups for the particle-particle channel of FL
    SGppL::Vector{SymmetryGroup}

    # symmetry groups for the particle-hole channels of FL
    SGphL::Vector{SymmetryGroup}

    # symmetry groups for the particle-particle channel of F0. Used only for fdPA
    SG0pp2::SymmetryGroup

    # symmetry groups for the particle-hole channels of F0. Used only for fdPA
    SG0ph2::SymmetryGroup

    # Parallelization mode
    mode::Symbol

    # Pre-evaluated vertices. To be used in the BSE of the K3 class
    # See build_K3_cache.jl for details.
    cache_Γpx :: MF_K3{Q}
    cache_F0p :: MF_K3{Q}
    cache_F0a :: MF_K3{Q}
    cache_F0t :: MF_K3{Q}
    cache_Γpp :: MF_K3{Q}
    cache_Γa  :: MF_K3{Q}
    cache_Γt  :: MF_K3{Q}
    cache_Fp  :: MF_K3{Q}
    cache_Fa  :: MF_K3{Q}
    cache_Ft  :: MF_K3{Q}

    # constructor
    function ParquetSolver(
        nK1   :: Int64,
        nK2   :: NTuple{2,Int64},
        nK3   :: NTuple{2,Int64},
        Gbare :: MF_G{Q},
        G0    :: MF_G{Q},
        Σ0    :: MF_G{Q},
        F0    :: RefVT,
              :: Type{VT} = Vertex,
        ;
        mode::Symbol = :threads,
    ) where {Q, VT, RefVT}

        T = MatsubaraFunctions.temperature(meshes(G0, Val(1)))

        # precompute bubbles for reference system
        mΠΩ = MatsubaraMesh(temperature(F0), nK1, Boson)
        mΠν = MatsubaraMesh(temperature(F0), nK1, Fermion)
        Π0pp = MeshFunction(mΠΩ, mΠν; data_t=Q)
        Π0ph = copy(Π0pp)

        bubbles!(Π0pp, Π0ph, G0)

        # single-particle Green's function and self-energy
        # Initialization: Σ = Σ0
        G = copy(G0)
        Σ = copy(Σ0)
        Dyson!(G, Σ, Gbare)

        # bubbles
        Πpp = copy(Π0pp)
        Πph = copy(Π0pp)
        bubbles!(Πpp, Πph, G)

        # channel-decomposed two-particle vertex
        F = VT(F0, T, nK1, nK2, nK3)
        Fbuff = Vertex(RefVertex(T, 0.0, Q), T, nK1, nK2, nK3)

        # Vertices for the SDE
        Lpp = copy(F.γp.K2)
        Lph = copy(Lpp)
        L0pp = copy(Lpp)
        L0ph = copy(Lpp)

        # symmetry groups
        SGΣ = SymmetryGroup(Σ)
        SGpp = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGph = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGppL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGphL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]

        # Symmetry group of F0, needed for the BSE of the reference vertex in SDE for fdPA
        SG0pp2 = SymmetryGroup(L0pp)
        SG0ph2 = SymmetryGroup(L0ph)

        # Check consistency of meshes
        @assert meshes(Gbare) == meshes(G0) == meshes(Σ0) == meshes(Σ) == meshes(Σ)

        # Pre-evaluated caches for the BSE of the K3 class
        cache_Γpx = copy(F.γp.K3)
        cache_F0p = copy(F.γp.K3)
        cache_F0a = copy(F.γp.K3)
        cache_F0t = copy(F.γp.K3)
        cache_Γpp = copy(F.γp.K3)
        cache_Γa  = copy(F.γp.K3)
        cache_Γt  = copy(F.γp.K3)
        cache_Fp  = copy(F.γp.K3)
        cache_Fa  = copy(F.γp.K3)
        cache_Ft  = copy(F.γp.K3)

        return new{Q, typeof(F), RefVT}(Gbare, G0, Π0pp, Π0ph, Σ0, F0, G, Πpp, Πph, Σ, F, Fbuff, copy(Fbuff),
        Lpp, Lph, L0pp, L0ph,
        SGΣ, SGpp, SGph, SGppL, SGphL, SG0pp2, SG0ph2, mode, cache_Γpx, cache_F0p, cache_F0a,
        cache_F0t, cache_Γpp, cache_Γa, cache_Γt, cache_Fp, cache_Fa, cache_Ft) :: ParquetSolver{Q}
    end
end


function Base.show(io::IO, S::AbstractSolver{Q}) where {Q}
    print(io, "$(nameof(typeof(S))){$Q}, U = $(real(bare_vertex(S.F0, aCh, pSp))), T = $(temperature(S))\n")
    print(io, "F0 K3 : $(numK3(S.F0))\n")
    print(io, "F  K1 : $(numK1(S.F))\n")
    print(io, "F  K2 : $(numK2(S.F))\n")
    print(io, "F  K3 : $(numK3(S.F))")
end


# Construct for parquet approximation
function parquet_solver_siam_parquet_approximation(
    nG::Int64,
    nK1::Int64,
    nK2::NTuple{2,Int64},
    nK3::NTuple{2,Int64},
    :: Type{Q} = ComplexF64,
    ;
    mode::Symbol = :threads,
    VT = Vertex,
    e,
    Δ,
    D,
    T,
    U,
) where {Q}

    # Mesh for the Green functions and self-energy
    mG = MatsubaraMesh(T, nG, Fermion)

    Gbare = siam_bare_Green(mG, Q; e, Δ, D)

    # Reference system: G0 = Σ0 = 0, F0 = U (parquet approximation)
    G0 = MeshFunction(mG; data_t = Q)
    Σ0 = MeshFunction(mG; data_t = Q)
    F0 = fdDGAsolver.RefVertex(T, U, Q)
    set!(G0, 0)
    set!(Σ0, 0)

    ParquetSolver(nK1, nK2, nK3, Gbare, G0, Σ0, F0, VT; mode)
end

# symmetry group initialization
function init_sym_grp!(
    S::ParquetSolver
)::Nothing

    # self-energy
    S.SGΣ = my_SymmetryGroup([Symmetry{1}(sΣ)], S.Σ)

    # particle-particle channel
    S.SGpp[1] = my_SymmetryGroup([Symmetry{1}(sK1pp)], S.F.γp.K1)
    S.SGpp[2] = my_SymmetryGroup([Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)], S.F.γp.K2)
    S.SGpp[3] = my_SymmetryGroup([Symmetry{3}(sK3pp1), Symmetry{3}(sK3pp2), Symmetry{3}(sK3pp3)], S.F.γp.K3)
    S.SGppL[1] = S.SGpp[1]
    S.SGppL[2] = S.SGpp[2]
    S.SGppL[3] = my_SymmetryGroup([Symmetry{3}(sK3pp1), Symmetry{3}(sK3pp3)], S.F.γp.K3)

    # particle-hole channels
    S.SGph[1] = my_SymmetryGroup([Symmetry{1}(sK1ph)], S.F.γt.K1)
    S.SGph[2] = my_SymmetryGroup([Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)], S.F.γt.K2)
    S.SGph[3] = my_SymmetryGroup([Symmetry{3}(sK3ph1), Symmetry{3}(sK3ph2), Symmetry{3}(sK3ph3)], S.F.γt.K3)
    S.SGphL[1] = S.SGph[1]
    S.SGphL[2] = S.SGph[2]
    S.SGphL[3] = my_SymmetryGroup([Symmetry{3}(sK3ph1), Symmetry{3}(sK3ph3)], S.F.γt.K3)

    # For F0
    S.SG0pp2 = my_SymmetryGroup([Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)], S.L0pp)
    S.SG0ph2 = my_SymmetryGroup([Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)], S.L0pp)

    return nothing
end

# getter methods
function MatsubaraFunctions.temperature(
    S::AbstractSolver
)::Float64

    return MatsubaraFunctions.temperature(S.F)
end

numG(S::AbstractSolver)::Int64 = N(meshes(S.G, Val(1)))
numΣ(S::AbstractSolver)::Int64 = N(meshes(S.Σ, Val(1)))
numK1(S::AbstractSolver)::Int64 = numK1(S.F)
numK2(S::AbstractSolver)::NTuple{2,Int64} = numK2(S.F)
numK3(S::AbstractSolver)::NTuple{2,Int64} = numK3(S.F)

# flattening and unflattening of solver
function MatsubaraFunctions.flatten(
    S::AbstractSolver{Q}
)::Vector{Q} where {Q}

    return vcat(flatten(S.F), flatten(S.Σ))
end

function MatsubaraFunctions.flatten!(
    S::AbstractSolver{Q},
    x::Vector{Q}
)::Nothing where {Q}

    lenΣ = length(S.Σ.data)
    flatten!(S.F, @view x[1:end-lenΣ])
    flatten!(S.Σ, @view x[end-lenΣ+1:end])

    return nothing
end

function MatsubaraFunctions.unflatten!(
    S::AbstractSolver{Q},
    x::Vector{Q}
)::Nothing where {Q}

    lenΣ = length(S.Σ.data)
    unflatten!(S.F, @view x[1:end-lenΣ])
    unflatten!(S.Σ, @view x[end-lenΣ+1:end])

    return nothing
end

# save solver to HDF5
function MatsubaraFunctions.save!(
    f::HDF5.File,
    label::String,
    S::AbstractSolver
)::Nothing

    MatsubaraFunctions.save!(f, "Gbare", S.Gbare)

    MatsubaraFunctions.save!(f, "G0", S.G0)
    MatsubaraFunctions.save!(f, "Σ0", S.Σ0)
    MatsubaraFunctions.save!(f, "F0", S.F0)
    MatsubaraFunctions.save!(f, "Π0pp", S.Π0pp)
    MatsubaraFunctions.save!(f, "Π0ph", S.Π0ph)

    MatsubaraFunctions.save!(f, "G", S.G)
    MatsubaraFunctions.save!(f, "Σ", S.Σ)
    MatsubaraFunctions.save!(f, "F", S.F)
    MatsubaraFunctions.save!(f, "Πpp", S.Πpp)
    MatsubaraFunctions.save!(f, "Πph", S.Πph)

    return nothing
end

function my_symmetrize!(S :: AbstractSolver)
    my_symmetrize!(S.F.γp.K1, S.SGpp[1])
    my_symmetrize!(S.F.γp.K2, S.SGpp[2])
    my_symmetrize!(S.F.γp.K3, S.SGpp[3])
    my_symmetrize!(S.F.γt.K1, S.SGph[1])
    my_symmetrize!(S.F.γt.K2, S.SGph[2])
    my_symmetrize!(S.F.γt.K3, S.SGph[3])
    my_symmetrize!(S.F.γa.K1, S.SGph[1])
    my_symmetrize!(S.F.γa.K2, S.SGph[2])
    my_symmetrize!(S.F.γa.K3, S.SGph[3])
end
