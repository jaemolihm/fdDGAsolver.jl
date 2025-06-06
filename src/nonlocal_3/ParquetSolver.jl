mutable struct NL3_ParquetSolver{Q, VT, RefVT} <: AbstractSolver{Q}
    # Bare Green function
    Gbare :: NL_MF_G{Q}

    # single-particle Green's function and bubbles for reference system
    G0::NL_MF_G{Q}
    Π0pp::NL2_MF_Π{Q}
    Π0ph::NL2_MF_Π{Q}

    # self-energy for reference system
    Σ0::NL_MF_G{Q}

    # two-particle vertex for reference system
    F0::RefVT

    # single-particle Green's function and bubbles for target system
    G::NL_MF_G{Q}
    Πpp::NL2_MF_Π{Q}
    Πph::NL2_MF_Π{Q}

    # self-energy for target system
    Σ::NL_MF_G{Q}

    # channel-decomposed two-particle vertex for target system
    F::VT

    # channel-decomposed two-particle vertex buffer
    Fbuff::NL3_Vertex{Q, RefVertex{Q}}

    # channel-decomposed two-particle vertex buffer for left part
    FL::NL3_Vertex{Q, RefVertex{Q}}

    # K2 vertices used in the SDE
    Lpp  :: NL2_MF_K2{Q}
    Lph  :: NL2_MF_K2{Q}
    L0pp :: NL2_MF_K2{Q}
    L0ph :: NL2_MF_K2{Q}

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
    cache_Γpx :: NL3_MF_K3{Q}
    cache_F0p :: NL3_MF_K3{Q}
    cache_F0a :: NL3_MF_K3{Q}
    cache_F0t :: NL3_MF_K3{Q}
    cache_Γpp :: NL3_MF_K3{Q}
    cache_Γa  :: NL3_MF_K3{Q}
    cache_Γt  :: NL3_MF_K3{Q}
    cache_Fp  :: NL3_MF_K3{Q}
    cache_Fa  :: NL3_MF_K3{Q}
    cache_Ft  :: NL3_MF_K3{Q}

    # constructor
    function NL3_ParquetSolver(
        nK1   :: Int64,
        nK2   :: NTuple{2,Int64},
        nK3   :: NTuple{2,Int64},
        mK_Γ  :: KMesh,
        Gbare :: NL_MF_G{Q},
        G0    :: NL_MF_G{Q},
        Σ0    :: NL_MF_G{Q},
        F0    :: RefVT,
              :: Type{VT} = NL3_Vertex,
        ;
        mode::Symbol = :threads,
    ) where {Q, VT, RefVT}

        T = MatsubaraFunctions.temperature(meshes(G0, Val(1)))

        # precompute bubbles for reference system
        mΠΩ = MatsubaraMesh(temperature(F0), nK1, Boson)
        mΠν = MatsubaraMesh(temperature(F0), nK1, Fermion)
        Π0pp = MeshFunction(mΠΩ, mΠν, mK_Γ, mK_Γ; data_t=Q)
        Π0ph = copy(Π0pp)

        bubbles_real_space!(Π0pp, Π0ph, G0)

        # single-particle Green's function and self-energy
        # Initialization: Σ = Σ0
        G = copy(G0)
        Σ = copy(Σ0)
        Dyson!(G, Σ, Gbare)

        # bubbles
        Πpp = copy(Π0pp)
        Πph = copy(Π0pp)
        bubbles_real_space!(Πpp, Πph, G)

        # channel-decomposed two-particle vertex
        F = VT(F0, T, nK1, nK2, nK3, mK_Γ)
        Fbuff = NL3_Vertex(RefVertex(T, 0.0, Q), T, nK1, nK2, nK3, mK_Γ)

        # Vertices for the SDE
        Lpp = copy(F.γp.K2)
        Lph  = copy(Lpp)
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
        cache_F0t, cache_Γpp, cache_Γa, cache_Γt, cache_Fp, cache_Fa, cache_Ft) :: NL3_ParquetSolver{Q}
    end
end

function Base.show(io::IO, S::NL3_ParquetSolver{Q}) where {Q}
    print(io, "$(nameof(typeof(S))){$Q}, U = $(real(bare_vertex(S.F0, aCh, pSp))), T = $(temperature(S))\n")
    print(io, "F0 K3 : $(numK3(S.F0))\n")
    print(io, "F  K1 : $(numK1(S.F))\n")
    print(io, "F  K2 : $(numK2(S.F))\n")
    print(io, "F  K3 : $(numK3(S.F))\n")
    print(io, "F  P  : $(numP(S.F))")
end


# Construct for parquet approximation
function parquet_solver_hubbard_parquet_approximation_NL3(
    nG::Int64,
    nK1::Int64,
    nK2::NTuple{2,Int64},
    nK3::NTuple{2,Int64},
    mK_G :: KMesh,
    mK_Γ :: KMesh,
    :: Type{Q} = ComplexF64,
    ;
    mode::Symbol = :threads,
    VT = NL3_Vertex,
    T,
    U,
    μ, t1, t2 = 0., t3 = 0.,
) where {Q}

    # Mesh for the Green functions and self-energy
    mG = MatsubaraMesh(T, nG, Fermion)

    Gbare = hubbard_bare_Green(mG, mK_G, Q; μ, t1, t2, t3)

    # Reference system: G0 = Σ0 = 0, F0 = U (parquet approximation)
    G0 = MeshFunction(mG, mK_G; data_t = Q)
    Σ0 = MeshFunction(mG, mK_G; data_t = Q)
    F0 = RefVertex(T, U, Q)
    set!(G0, 0)
    set!(Σ0, 0)

    NL3_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, G0, Σ0, F0, VT; mode)
end


# symmetry group initialization
function init_sym_grp!(
    S::NL3_ParquetSolver
    )::Nothing

    mK_Σ = meshes(S.Σ, Val(2))
    mK_Γ = meshes(S.F.γp.K1, Val(2))

    # self-energy
    @time S.SGΣ = my_SymmetryGroup(Symmetry{2}[
        Symmetry{2}(w -> sΣ_conj(w, mK_Σ)),
        Symmetry{2}(w -> sΣ_ref(w, mK_Σ)),
        Symmetry{2}(w -> sΣ_rot(w, mK_Σ))
    ], S.Σ);

    # Vertices in the particle-particle channel

    @time S.SGpp[1] = my_SymmetryGroup(Symmetry{2}[
        Symmetry{2}(w -> sK1_conj(w, mK_Γ)),
        Symmetry{2}(w -> sK1_ref(w, mK_Γ)),
        Symmetry{2}(w -> sK1_rot(w, mK_Γ)),
    ], S.F.γp.K1);

    @time S.SGpp[2] = my_SymmetryGroup([
        Symmetry{4}(w -> sK2_NL2_pp1(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_pp2(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_rot(w, mK_Γ)),
    ], S.F.γp.K2)

    @time S.SGpp[3] = my_SymmetryGroup([
        Symmetry{6}(w -> sK3_NL3_pp1(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_pp2(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_pp3(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ref(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_rot(w, mK_Γ)),
    ], S.F.γp.K3)

    S.SGppL[1] = S.SGpp[1]
    S.SGppL[2] = S.SGpp[2]
    @time S.SGppL[3] = my_SymmetryGroup([
        Symmetry{6}(w -> sK3_NL3_pp1(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_pp3(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ref(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_rot(w, mK_Γ)),
    ], S.F.γp.K3)

    # particle-hole channels

    S.SGph[1] = S.SGpp[1]

    @time S.SGph[2] = my_SymmetryGroup([
        Symmetry{4}(w -> sK2_NL2_ph1(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ph2(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_rot(w, mK_Γ)),
    ], S.F.γt.K2)

    @time S.SGph[3] = my_SymmetryGroup([
        Symmetry{6}(w -> sK3_NL3_ph1(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ph2(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ph3(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ref(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_rot(w, mK_Γ)),
    ], S.F.γt.K3)

    S.SGphL[1] = S.SGph[1]
    S.SGphL[2] = S.SGph[2]
    @time S.SGphL[3] = my_SymmetryGroup([
        Symmetry{6}(w -> sK3_NL3_ph1(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ph3(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ref(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_rot(w, mK_Γ)),
    ], S.F.γt.K3)


    # Symmetry group of F0, needed for the BSE of the reference vertex in SDE for fdPA
    @time S.SG0pp2 = my_SymmetryGroup([
        Symmetry{4}(w -> sK2_NL2_pp1(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_pp2(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_rot(w, mK_Γ)),
    ], S.L0pp)
    @time S.SG0ph2 = my_SymmetryGroup([
        Symmetry{4}(w -> sK2_NL2_ph1(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ph2(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_rot(w, mK_Γ)),
    ], S.L0pp)


    return nothing
end

# getter methods (some of them is defined for AbstractSolver)
numP_G(S::NL3_ParquetSolver)::Int64 = length(meshes(S.G, Val(2)))
numP_Γ(S::NL3_ParquetSolver)::Int64 = numP(S.F)

function bubbles!(S :: NL3_ParquetSolver)
    bubbles_real_space!(S)
    # bubbles_momentum_space!(S)
end
