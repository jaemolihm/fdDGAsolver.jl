using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using LinearAlgebra
using HDF5
using MatsubaraFunctions: mesh_index
using fdDGAsolver: numP_Γ, k0, kSW
using fdDGAsolver: sK1pp, sK2pp1, sK2pp2, sK3pp1, sK3pp2, sK3pp3, my_SymmetryGroup, sK1ph, sK2ph1, sK2ph2, sK3ph1, sK3ph2, sK3ph3

# JML note
# `julia -t 8 jun_Wu_point.jl 6 6 2` takes a few minutes per nlsolve iteration.
# It involves ~200 mfRG iterations and a single fdDGammaA iteration.
# To directly call these functions, use the following.
# @time fdDGAsolver.mfRGLinearMap(S) * flatten(S.F); # mfRG iteration
# @time iterate_solver!(S; strategy = :fdPA, update_Σ = false); # fdDGammaA iteration

"""
- `nmax` : Frequency box size
- `nq` : Momentum grid size
- `nl_method` : 1 for s-wave, 2 for K2(k,q)
- `filename_log` : Prefix of the file name. Files `filename_log.iter#.h5` will be created
                   for every self-energy iteration.
- `auto_restart` : If true, restart from the last log file.
"""
function solve(nmax, nq, nl_method; filename_log = nothing, auto_restart = true)
    mpi_ismain() && println("Solve Wu point, nmax = $nmax, nq = $nq, NL $nl_method")

    # --------------------------------------------------------------------------------
    # System parameters
    T = 0.2
    U = 5.6
    μ = 2.1800201007694464 - U/2
    t1 = 1.0
    t2 = -0.3

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    # --------------------------------------------------------------------------------
    # Load impurity vertex
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta5.0_t-1.0_U5.6_mu2.1800201007694464_numc1_numk255"
    data_triqs = load_vertex_from_triqs(prefix, T, U; half_filling = false)

    # --------------------------------------------------------------------------------
    # Symmetrize impurity vertex
    Γ = data_triqs.Γ
    # particle-particle channel
    SGpp1 = my_SymmetryGroup([Symmetry{1}(sK1pp)], Γ.γp.K1)
    SGpp2 = my_SymmetryGroup([Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)], Γ.γp.K2)
    SGpp3 = my_SymmetryGroup([Symmetry{3}(sK3pp1), Symmetry{3}(sK3pp2), Symmetry{3}(sK3pp3)], Γ.F0.Fp_p)
    # particle-hole channels
    SGph1 = my_SymmetryGroup([Symmetry{1}(sK1ph)], Γ.γt.K1)
    SGph2 = my_SymmetryGroup([Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)], Γ.γt.K2)
    SGph3 = my_SymmetryGroup([Symmetry{3}(sK3ph1), Symmetry{3}(sK3ph2), Symmetry{3}(sK3ph3)], Γ.F0.Ft_p)

    SGpp1(Γ.γp.K1)
    SGpp2(Γ.γp.K2)
    SGph1(Γ.γt.K1)
    SGph1(Γ.γa.K1)
    SGph2(Γ.γt.K2)
    SGph2(Γ.γa.K2)
    SGpp3(Γ.F0.Fp_p)
    SGpp3(Γ.F0.Fp_x)
    SGph3(Γ.F0.Ft_p)
    SGph3(Γ.F0.Ft_x)

    # --------------------------------------------------------------------------------
    # Solver parameters
    nG  = 4nmax
    nK1 = 4nmax
    nK2 = (nmax + 1, nmax)
    nK3 = (nmax + 1, nmax)

    mK_G = BrillouinZoneMesh(BrillouinZone(48, k1, k2))
    mK_Γ = BrillouinZoneMesh(BrillouinZone(nq, k1, k2))

    # --------------------------------------------------------------------------------

    # Set reference Green function and self-energy
    mG = MatsubaraMesh(T, nG, Fermion)
    Gbare = hubbard_bare_Green(mG, mK_G; μ, t1, t2)
    G0 = copy(Gbare)
    Σ0 = copy(Gbare)
    set!(G0, 0)
    set!(Σ0, 0)
    for ν in meshes(G0, Val(1))
        view(G0, ν, :) .= data_triqs.G(value(ν))
        view(Σ0, ν, :) .= data_triqs.Σ(value(ν))
    end

    # --------------------------------------------------------------------------------
    # Sanity check: impurity Dyson equation
    G_from_Dyson = copy(Gbare)
    Dyson!(G_from_Dyson, Σ0, Gbare)
    νs = value.(data_triqs.G.meshes[1])
    mpi_ismain() && @info "G_TRIQS - G_Dyson = ", norm(data_triqs.G.(νs) .- G_from_Dyson.(νs, Ref(kSW)))

    occ_target = compute_occupation(G_from_Dyson)
    mpi_ismain() && @info "occ_G_TRIQS - occ_G_Dyson = ", compute_occupation(data_triqs.G) - compute_occupation(G_from_Dyson)

    # --------------------------------------------------------------------------------
    # Initialize ParquetSolver
    if nl_method == 1
        F0 = NL_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mK_Γ)
        S = NL_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, copy(G0), copy(Σ0), F0; mode = :hybrid)
    else
        F0 = NL2_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mK_Γ)
        S = NL2_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, copy(G0), copy(Σ0), F0; mode = :hybrid)
    end

    init_sym_grp!(S)
    mpi_ismain() && @info "size of S = $(Base.summarysize(S) / 1e9) GB"

    # --------------------------------------------------------------------------------
    # Run mfRG solver

    fdDGAsolver.solve_using_mfRG_mix_G!(S; filename_log, maxiter = 200, occ_target, hubbard_params = (; t1, t2), mixing_G_init = 0.2, tol = 2e-4, auto_restart)


    return S
end

nmax = parse(Int, ARGS[1])
nq   = parse(Int, ARGS[2])
nl_method = parse(Int, ARGS[3])

# filename_log = "/globalscratch/ucl/modl/jmlihm/temp/Wu.mix_G.NL$nl_method.nmax$nmax.nq$nq"
filename_log = nothing

S = solve(nmax, nq, nl_method; filename_log);
