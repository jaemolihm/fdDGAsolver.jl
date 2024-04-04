mutable struct ParquetSolver{Q, RefVT}
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
    F::Vertex{Q, RefVT}

    # channel-decomposed two-particle vertex buffer
    Fbuff::Vertex{Q, RefVertex{Q}}

    # channel-decomposed two-particle vertex buffer for left part
    FL::Vertex{Q, RefVertex{Q}}

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
    function ParquetSolver(
        nG::Int64,
        nΣ::Int64,
        nK1::Int64,
        nK2::NTuple{2,Int64},
        nK3::NTuple{2,Int64},
        Gbare::MF_G{Q},
        G0::MF_G{Q},
        Σ0::MF_G{Q},
        F0::RefVT,
        ;
        mode::Symbol = :serial,
    ) where {Q, RefVT}

        T = MatsubaraFunctions.temperature(meshes(G0, 1))

        # precompute bubbles for reference system
        mΠΩ = MatsubaraMesh(temperature(F0), nK1, Boson)
        mΠν = MatsubaraMesh(temperature(F0), 2 * nK1, Fermion)
        Π0pp = MeshFunction(mΠΩ, mΠν; data_t=Q)
        Π0ph = copy(Π0pp)

        for Ω in value.(meshes(Π0pp, 1)), ν in value.(meshes(Π0pp, 2))
            Π0pp[Ω, ν] = G0(ν) * G0(Ω - ν)
            Π0ph[Ω, ν] = G0(Ω + ν) * G0(ν)
        end

        # single-particle Green's function and self-energy
        G = MeshFunction(MatsubaraMesh(T, nG, Fermion); data_t=Q)
        Σ = MeshFunction(MatsubaraMesh(T, nΣ, Fermion); data_t=Q)
        set!(G, 0)
        set!(Σ, 0)

        # bubbles
        Πpp = copy(Π0pp)
        Πph = copy(Π0pp)

        # channel-decomposed two-particle vertex
        F = Vertex(F0, T, nK1, nK2, nK3)
        Fbuff = Vertex(RefVertex(T, 0.0, Q), T, nK1, nK2, nK3)

        # symmetry groups
        SGΣ = SymmetryGroup(Σ)
        SGpp = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGph = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGppL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGphL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]

        # Symmetry group of F0, needed for the BSE of the reference vertex in SDE for fdPA
        if F0 isa Vertex
            SG0pp2 = SymmetryGroup(F0.γp.K2)
            SG0ph2 = SymmetryGroup(F0.γp.K2)
        else
            SG0pp2 = nothing
            SG0ph2 = nothing
        end

        return new{Q, RefVT}(Gbare, G0, Π0pp, Π0ph, Σ0, F0, G, Πpp, Πph, Σ, F, Fbuff, copy(Fbuff), SGΣ, SGpp, SGph, SGppL, SGphL, SG0pp2, SG0ph2, mode)::ParquetSolver{Q}
    end
end

Base.eltype(::Type{<:ParquetSolver{Q}}) where {Q} = Q

function Base.show(io::IO, S::ParquetSolver{Q}) where {Q}
    print(io, "$(nameof(typeof(S))){$Q}, U = $(real(bare_vertex(S.F0, aCh, pSp))), T = $(temperature(S))\n")
    print(io, "F0 K3 : $(numK3(S.F0))\n")
    print(io, "F  K1 : $(numK1(S.F))\n")
    print(io, "F  K2 : $(numK2(S.F))\n")
    print(io, "F  K3 : $(numK3(S.F))")
end


# Construct for parquet approximation
function parquet_solver_siam_parquet_approximation(
    nG::Int64,
    nΣ::Int64,
    nK1::Int64,
    nK2::NTuple{2,Int64},
    nK3::NTuple{2,Int64},
    :: Type{Q} = ComplexF64,
    ;
    mode::Symbol = :serial,
    e,
    Δ,
    D,
    T,
    U,
) where {Q}

    # Mesh for the Green functions and self-energy
    mG = MatsubaraMesh(T, nG, Fermion)
    mΣ = MatsubaraMesh(T, nΣ, Fermion)

    Gbare = siam_bare_Green(mG, Q; e, Δ, D)

    # Reference system: G0 = Σ0 = 0, F0 = U (parquet approximation)
    G0 = MeshFunction(mG; data_t = Q)
    Σ0 = MeshFunction(mΣ; data_t = Q)
    F0 = fdDGAsolver.RefVertex(T, U, Q)
    set!(G0, 0)
    set!(Σ0, 0)

    ParquetSolver(nG, nΣ, nK1, nK2, nK3, Gbare, G0, Σ0, F0; mode)
end

# construct from CTINT input data
function ParquetSolver(
    data::HDF5.File,
    T::Float64,
    U::Float64,
    numG::Int64,
    numΣ::Int64,
    numK1::Int64,
    numK2::NTuple{2,Int64},
    numK3::NTuple{2,Int64},
    ::Type{Q}=ComplexF64,
) where {Q}

    println("Loading reference system ...")

    # load G
    Gdat = read(data, "G")
    Gidx = read(data, "G_last_idx")
    gG = MatsubaraMesh(T, Gidx + 1, Fermion)
    @assert typeof(Gdat) == Vector{Q}
    @assert last_index(gG) == Gidx
    Garr = Array{Q,2}(undef, length(Gdat), 1)
    Garr[:] .= Gdat
    G = MeshFunction(gG, Garr)

    # load Σ
    Σdat = read(data, "Σ")
    Σidx = read(data, "Σ_last_idx")
    gΣ = MatsubaraMesh(T, Σidx + 1, Fermion)
    @assert typeof(Σdat) == Vector{Q}
    @assert last_index(gΣ) == Σidx
    Σarr = Array{Q,2}(undef, length(Σdat), 1)
    Σarr[:] .= Σdat
    Σ = MeshFunction(gΣ, Σarr)

    # load χ2
    χ2dat = permutedims(read(data, "χ2pp_p"), (2, 1))[:, 1]
    χ2idx = read(data, "χ2_last_idx")
    gχ2 = MatsubaraMesh(T, χ2idx + 1, Boson)
    @assert typeof(χ2dat) == Vector{Q}
    @assert last_index(gχ2) == χ2idx
    χ2arr = Array{Q,2}(undef, length(χ2dat), 1)
    χ2arr[:] .= χ2dat
    χ2 = MeshFunction(g2, χ2arr)

    # load χ3
    χ3dat = permutedims(read(data, "χ3pp_p"), (3, 2, 1))[:, :, 1]
    χ3idxΩ = read(data, "χ3_last_idx_Ω")
    χ3idxν = read(data, "χ3_last_idx_ν")
    gχ3Ω = MatsubaraMesh(T, χ3idxΩ + 1, Boson)
    gχ3ν = MatsubaraMesh(T, χ3idxν + 1, Fermion)
    @assert typeof(χ3dat) == Matrix{Q}
    @assert last_index(gχ3Ω) == χ3idxΩ
    @assert last_index(gχ3ν) == χ3idxν
    χ3arr = Array{Q,3}(undef, size(χ3dat, 1), size(χ3dat, 2), 1)
    χ3arr[:, :, 1] .= χ3dat
    χ3 = MeshFunction(gχ3Ω, gχ3ν, χ3arr)

    # load F
    Fp_p_dat = permutedims(read(data, "Fpp_p"), (4, 3, 2, 1))[:, :, :, 1]
    Fp_x_dat = permutedims(read(data, "Fpp_x"), (4, 3, 2, 1))[:, :, :, 1]
    Ft_p_dat = permutedims(read(data, "Fph_p"), (4, 3, 2, 1))[:, :, :, 1]
    Ft_x_dat = permutedims(read(data, "Fph_x"), (4, 3, 2, 1))[:, :, :, 1]
    FidxΩ = read(data, "F_last_idx_Ω")
    Fidxν = read(data, "F_last_idx_ν")
    gFΩ = MatsubaraMesh(T, FidxΩ + 1, Boson)
    gFν = MatsubaraMesh(T, Fidxν + 1, Fermion)
    @assert typeof(Fp_p_dat) == Array{Q,3}
    @assert typeof(Fp_x_dat) == Array{Q,3}
    @assert typeof(Ft_p_dat) == Array{Q,3}
    @assert typeof(Ft_x_dat) == Array{Q,3}
    @assert typeof(χ3dat) == Matrix{Q}
    @assert last_index(gFΩ) == FidxΩ
    @assert last_index(gFν) == Fidxν
    Fp_p_arr = Array{Q,4}(undef, size(Fp_p_dat, 1), size(Fp_p_dat, 2), size(Fp_p_dat, 3), 1)
    Fp_x_arr = deepcopy(Fp_p_arr)
    Ft_p_arr = deepcopy(Fp_p_arr)
    Ft_x_arr = deepcopy(Fp_p_arr)
    Fp_p_arr[:, :, :, 1] .= Fp_p_dat
    Fp_x_arr[:, :, :, 1] .= Fp_x_dat
    Ft_p_arr[:, :, :, 1] .= Ft_p_dat
    Ft_x_arr[:, :, :, 1] .= Ft_x_dat
    Fp_p = MeshFunction(gFΩ, gFν, gFν, Fp_p_arr)
    Fp_x = MeshFunction(gFΩ, gFν, gFν, Fp_x_arr)
    Ft_p = MeshFunction(gFΩ, gFν, gFν, Ft_p_arr)
    Ft_x = MeshFunction(gFΩ, gFν, gFν, Ft_x_arr)

    # from χ2 compute K1
    K1 = deepcopy(χ2)
    mult!(K1, -U * U)

    # from χ3 and K1 compute K2
    K2 = deepcopy(χ3)

    for Ω in grids(K2, 1), ν in grids(K2, 2)
        # minus sign because we factor out (-i) from G
        K2[Ω, ν] = -U * χ3(Ω, ν) / G(ν) / G(Ω - ν) - K1(Ω) - U
    end

    println("Building target system ...")
    return ParquetSolver(numG, numΣ, numK1, numK2, numK3, G, Σ, RefVertex(U, Fp_p, Fp_x, Ft_p, Ft_x))::ParquetSolver{Q}
end

# symmetry group initialization
function init_sym_grp!(
    S::ParquetSolver
)::Nothing

    # self-energy
    S.SGΣ = SymmetryGroup([Symmetry{1}(sΣ)], S.Σ)

    # particle-particle channel
    S.SGpp[1] = SymmetryGroup([Symmetry{1}(sK1pp)], S.F.γp.K1)
    S.SGpp[2] = SymmetryGroup([Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)], S.F.γp.K2)
    S.SGpp[3] = SymmetryGroup([Symmetry{3}(sK3pp1), Symmetry{3}(sK3pp2), Symmetry{3}(sK3pp3)], S.F.γp.K3)
    S.SGppL[1] = S.SGpp[1]
    S.SGppL[2] = S.SGpp[2]
    S.SGppL[3] = SymmetryGroup([Symmetry{3}(sK3pp1), Symmetry{3}(sK3pp3)], S.F.γp.K3)

    # particle-hole channels
    S.SGph[1] = SymmetryGroup([Symmetry{1}(sK1ph)], S.F.γt.K1)
    S.SGph[2] = SymmetryGroup([Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)], S.F.γt.K2)
    S.SGph[3] = SymmetryGroup([Symmetry{3}(sK3ph1), Symmetry{3}(sK3ph2), Symmetry{3}(sK3ph3)], S.F.γt.K3)
    S.SGphL[1] = S.SGph[1]
    S.SGphL[2] = S.SGph[2]
    S.SGphL[3] = SymmetryGroup([Symmetry{3}(sK3ph1), Symmetry{3}(sK3ph3)], S.F.γt.K3)

    # For F0
    if S.SG0pp2 !== nothing
        S.SG0pp2 = SymmetryGroup([Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)], S.F0.γp.K2)
        S.SG0ph2 = SymmetryGroup([Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)], S.F0.γt.K2)
    end

    return nothing
end

# getter methods
function MatsubaraFunctions.temperature(
    S::ParquetSolver
)::Float64

    return MatsubaraFunctions.temperature(S.F)
end

numG(S::ParquetSolver)::Int64 = N(grids(S.G, 1))
numΣ(S::ParquetSolver)::Int64 = N(grids(S.Σ, 1))
numK1(S::ParquetSolver)::Int64 = numK1(S.F)
numK2(S::ParquetSolver)::NTuple{2,Int64} = numK2(S.F)
numK3(S::ParquetSolver)::NTuple{2,Int64} = numK3(S.F)

# flattening and unflattening of solver
function MatsubaraFunctions.flatten(
    S::ParquetSolver{Q}
)::Vector{Q} where {Q}

    return vcat(flatten(S.F), flatten(S.Σ))
end

function MatsubaraFunctions.flatten!(
    S::ParquetSolver{Q},
    x::Vector{Q}
)::Nothing where {Q}

    lenΣ = length(S.Σ.data)
    flatten!(S.F, @view x[1:end-lenΣ])
    flatten!(S.Σ, @view x[end-lenΣ+1:end])

    return nothing
end

function MatsubaraFunctions.unflatten!(
    S::ParquetSolver{Q},
    x::Vector{Q}
)::Nothing where {Q}

    lenΣ = length(S.Σ.data)
    unflatten!(S.F, @view x[1:end-lenΣ])
    unflatten!(S.Σ, @view x[end-lenΣ+1:end])

    return nothing
end
