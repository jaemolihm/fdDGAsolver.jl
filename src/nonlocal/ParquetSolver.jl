mutable struct NL_ParquetSolver{Q, RefVT} <: AbstractSolver{Q}
    # Bare Green function
    Gbare :: NL_MF_G{Q}

    # single-particle Green's function and bubbles for reference system
    G0::NL_MF_G{Q}
    Π0pp::NL_MF_Π{Q}
    Π0ph::NL_MF_Π{Q}

    # self-energy for reference system
    Σ0::NL_MF_G{Q}

    # two-particle vertex for reference system
    F0::RefVT

    # single-particle Green's function and bubbles for target system
    G::NL_MF_G{Q}
    Πpp::NL_MF_Π{Q}
    Πph::NL_MF_Π{Q}

    # self-energy for target system
    Σ::NL_MF_G{Q}

    # channel-decomposed two-particle vertex for target system
    F::NL_Vertex{Q, RefVT}

    # channel-decomposed two-particle vertex buffer
    Fbuff::NL_Vertex{Q, RefVertex{Q}}

    # channel-decomposed two-particle vertex buffer for left part
    FL::NL_Vertex{Q, RefVertex{Q}}

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
    SG0pp2::Union{SymmetryGroup, Nothing}

    # symmetry groups for the particle-hole channels of F0. Used only for fdPA
    SG0ph2::Union{SymmetryGroup, Nothing}

    # Parallelization mode
    mode::Symbol

    # constructor
    function NL_ParquetSolver(
        nG    :: Int64,
        nΣ    :: Int64,
        nK1   :: Int64,
        nK2   :: NTuple{2,Int64},
        nK3   :: NTuple{2,Int64},
        mK_Γ  :: KMesh,
        Gbare :: NL_MF_G{Q},
        G0    :: NL_MF_G{Q},
        Σ0    :: NL_MF_G{Q},
        F0    :: RefVT,
        ;
        mode::Symbol = :serial,
    ) where {Q, RefVT}

        T = MatsubaraFunctions.temperature(meshes(G0, 1))

        # precompute bubbles for reference system
        mΠΩ = MatsubaraMesh(temperature(F0), nK1, Boson)
        mΠν = MatsubaraMesh(temperature(F0), 2 * nK1, Fermion)
        Π0pp = MeshFunction(mΠΩ, mΠν, mK_Γ, mK_Γ; data_t=Q)
        Π0ph = copy(Π0pp)

        bubbles!(Π0pp, Π0ph, G0)

        # single-particle Green's function and self-energy
        # The self-energy has the same momentum resolution as the vertex mk_Γ,
        # which is coarser than that of the bare and full Green functions mk_G.
        # Initialization: G = Gbare, Σ = 0
        mK_G = meshes(Gbare, 2)
        G = MeshFunction(MatsubaraMesh(T, nG, Fermion), mK_G; data_t = Q)
        Σ = MeshFunction(MatsubaraMesh(T, nΣ, Fermion), mK_Γ; data_t = Q)
        set!(G, Gbare)
        set!(Σ, 0)

        # bubbles
        Πpp = copy(Π0pp)
        Πph = copy(Π0pp)
        bubbles!(Πpp, Πph, G)

        # channel-decomposed two-particle vertex
        F = NL_Vertex(F0, T, nK1, nK2, nK3, mK_Γ)
        Fbuff = NL_Vertex(RefVertex(T, 0.0, Q), T, nK1, nK2, nK3, mK_Γ)

        # symmetry groups
        SGΣ = SymmetryGroup(Σ)
        SGpp = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGph = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGppL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGphL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]

        # Symmetry group of F0, needed for the BSE of the reference vertex in SDE for fdPA
        if F0 isa NL_Vertex
            SG0pp2 = SymmetryGroup(F0.γp.K2)
            SG0ph2 = SymmetryGroup(F0.γp.K2)
        else
            SG0pp2 = nothing
            SG0ph2 = nothing
        end

        return new{Q, RefVT}(Gbare, G0, Π0pp, Π0ph, Σ0, F0, G, Πpp, Πph, Σ, F, Fbuff, copy(Fbuff), SGΣ, SGpp, SGph, SGppL, SGphL, SG0pp2, SG0ph2, mode)::NL_ParquetSolver{Q}
    end
end

function Base.show(io::IO, S::NL_ParquetSolver{Q}) where {Q}
    print(io, "$(nameof(typeof(S))){$Q}, U = $(real(bare_vertex(S.F0, aCh, pSp))), T = $(temperature(S))\n")
    print(io, "F0 K3 : $(numK3(S.F0))\n")
    print(io, "F  K1 : $(numK1(S.F))\n")
    print(io, "F  K2 : $(numK2(S.F))\n")
    print(io, "F  K3 : $(numK3(S.F))\n")
    print(io, "F  P  : $(numP(S.F))")
end


# Construct for parquet approximation
function parquet_solver_hubbard_parquet_approximation(
    nG::Int64,
    nΣ::Int64,
    nK1::Int64,
    nK2::NTuple{2,Int64},
    nK3::NTuple{2,Int64},
    mK_G :: KMesh,
    mK_Γ :: KMesh,
    :: Type{Q} = ComplexF64,
    ;
    mode::Symbol = :serial,
    T,
    U,
    μ, t1, t2 = 0, t3 = 0,
) where {Q}

    # Mesh for the Green functions and self-energy
    mG = MatsubaraMesh(T, nG, Fermion)
    mΣ = MatsubaraMesh(T, nΣ, Fermion)

    Gbare = hubbard_bare_Green(mG, mK_G, Q; μ, t1, t2, t3)

    # Reference system: G0 = Σ0 = 0, F0 = U (parquet approximation)
    G0 = MeshFunction(mG, mK_G; data_t = Q)
    Σ0 = MeshFunction(mΣ, mK_Γ; data_t = Q)
    F0 = RefVertex(T, U, Q)
    set!(G0, 0)
    set!(Σ0, 0)

    NL_ParquetSolver(nG, nΣ, nK1, nK2, nK3, mK_Γ, Gbare, G0, Σ0, F0; mode)
end


# symmetry group initialization
function init_sym_grp!(
    S::NL_ParquetSolver
)::Nothing

    mK_Σ = meshes(S.Σ, 2)
    mK_Γ = meshes(S.F.γp.K1, 2)

    # self-energy
    S.SGΣ = SymmetryGroup(Symmetry{2}[
        Symmetry{2}(w -> sΣ_conj(w,)),
        Symmetry{2}(w -> sΣ_ref(w, mK_Σ)),
        Symmetry{2}(w -> sΣ_rot(w, mK_Σ))
    ], S.Σ);

    # Vertices in the particle-particle channel

    S.SGpp[1] = SymmetryGroup(Symmetry{2}[
        Symmetry{2}(w -> sK1_conj(w,)),
        Symmetry{2}(w -> sK1_ref(w, mK_Γ)),
        Symmetry{2}(w -> sK1_rot(w, mK_Γ)),
    ], S.F.γp.K1);

    S.SGpp[2] = SymmetryGroup([
        Symmetry{3}(w -> sK2pp1( w, mK_Γ)),
        Symmetry{3}(w -> sK2pp2( w, mK_Γ)),
        Symmetry{3}(w -> sK2_ref(w, mK_Γ)),
        Symmetry{3}(w -> sK2_rot(w, mK_Γ)),
    ], S.F.γp.K2)

    S.SGpp[3] = SymmetryGroup([
        Symmetry{4}(w -> sK3pp1( w, mK_Γ)),
        Symmetry{4}(w -> sK3pp2( w, mK_Γ)),
        Symmetry{4}(w -> sK3pp3( w, mK_Γ)),
        Symmetry{4}(w -> sK3_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK3_rot(w, mK_Γ)),
    ], S.F.γp.K3)

    S.SGppL[1] = S.SGpp[1]
    S.SGppL[2] = S.SGpp[2]
    S.SGppL[3] = SymmetryGroup([
        Symmetry{4}(w -> sK3pp1( w, mK_Γ)),
        Symmetry{4}(w -> sK3pp3( w, mK_Γ)),
        Symmetry{4}(w -> sK3_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK3_rot(w, mK_Γ)),
    ], S.F.γp.K3)

    # particle-hole channels

    S.SGph[1] = S.SGpp[1]

    S.SGph[2] = SymmetryGroup([
        Symmetry{3}(w -> sK2ph1( w, mK_Γ)),
        Symmetry{3}(w -> sK2ph2( w, mK_Γ)),
        Symmetry{3}(w -> sK2_ref(w, mK_Γ)),
        Symmetry{3}(w -> sK2_rot(w, mK_Γ)),
    ], S.F.γt.K2)

    S.SGph[3] = SymmetryGroup([
        Symmetry{4}(w -> sK3ph1( w, mK_Γ)),
        Symmetry{4}(w -> sK3ph2( w, mK_Γ)),
        Symmetry{4}(w -> sK3ph3( w, mK_Γ)),
        Symmetry{4}(w -> sK3_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK3_rot(w, mK_Γ)),
    ], S.F.γt.K3)

    S.SGphL[1] = S.SGph[1]
    S.SGphL[2] = S.SGph[2]
    S.SGphL[3] = SymmetryGroup([
        Symmetry{4}(w -> sK3ph1( w, mK_Γ)),
        Symmetry{4}(w -> sK3ph3( w, mK_Γ)),
        Symmetry{4}(w -> sK3_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK3_rot(w, mK_Γ)),
    ], S.F.γt.K3)

    # # For F0
    # if S.SG0pp2 !== nothing
    #     S.SG0pp2 = SymmetryGroup([Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)], S.F0.γp.K2)
    #     S.SG0ph2 = SymmetryGroup([Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)], S.F0.γt.K2)
    # end

    return nothing
end

# getter methods (some of them is defined for AbstractSolver)
numP_G(S::NL_ParquetSolver)::Int64 = length(meshes(S.G, 2))
numP_Γ(S::NL_ParquetSolver)::Int64 = numP(S.F)
