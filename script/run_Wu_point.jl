using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using LinearAlgebra
using HDF5
using fdDGAsolver: kSW

# JML note
# `julia -t 8 jun_Wu_point.jl 6 6 2` takes a few minutes per nlsolve iteration.
# It involves ~200 mfRG iterations and a single fdDGammaA iteration.
# To directly call these functions, use the following.
# @time fdDGAsolver.mfRGLinearMap(S) * flatten(S.F); # mfRG iteration
# @time iterate_solver!(S; strategy = :fdPA, update_Σ = false); # fdDGammaA iteration

"""
- `nmax` : Frequency box size
- `nq` : Momentum grid size
- `nl_method` : 1 for s-wave, 2 for K2(k,q), -2 for SBE(k,q) + s-wave MBE
- `filename_log` : Prefix of the file name. Files `filename_log.iter#.h5` will be created
                   for every self-energy iteration.
- `auto_restart` : If true, restart from the last log file.
"""
function solve(nmax, nq, nl_method; filename_log = nothing, auto_restart = true, tol = 1e-3)
    if mpi_ismain()
        println("Solve Wu point, nmax = $nmax, nq = $nq, NL $nl_method")
        if nl_method == 1
            println("s-wave with asymptotic decomposition")
        elseif nl_method == 2
            println("Asymptotic decomposition with K1(q), K2(k, q) and s-wave K3")
        elseif nl_method == -2
            println("MBE with K1(q), K2(k, q) and s-wave M")
        end
    end

    # --------------------------------------------------------------------------------
    # System parameters
    T = 0.2
    U = 5.6
    μ = 2.1800201007694464 - U/2
    t1 = 1.0
    t2 = -0.3

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    data_triqs = load_triqs_data("/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point.h5")

    # T = 0.5
    # U = 2.089
    # μ = 0.
    # t1 = -0.25
    # t2 = 0.
    # data_triqs = load_triqs_data("/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/high_temperature_U2.089.h5")
    # --------------------------------------------------------------------------------
    # Solver parameters
    nG  = 4nmax
    nK1 = 4nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

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

    occ_target = data_triqs.occ
    mpi_ismain() && @info "occ_G_TRIQS - occ_G_Dyson = ", compute_occupation(data_triqs.G) - compute_occupation(G_from_Dyson)

    # --------------------------------------------------------------------------------
    # Initialize ParquetSolver
    if nl_method == 1
        F0 = NL_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mK_Γ)
        S = NL_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, copy(G0), copy(Σ0), F0; mode = :hybrid)
    elseif nl_method == 2
        F0 = NL2_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mK_Γ)
        S = NL2_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, copy(G0), copy(Σ0), F0; mode = :hybrid)
    elseif nl_method == -2
        F0 = NL2_MBEVertex(asymptotic_to_mbe(data_triqs.Γ), T, nK1, nK2, nK3, mK_Γ)
        S = NL2_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, copy(G0), copy(Σ0), F0, NL2_MBEVertex; mode = :hybrid)
    else
        throw(ArgumentError("Invalid nl_method: $nl_method"))
    end

    init_sym_grp!(S)
    mpi_ismain() && @info "size of S = $(Base.summarysize(S) / 1e9) GB"

    # --------------------------------------------------------------------------------
    # Run mfRG solver

    fdDGAsolver.solve_using_mfRG!(S; filename_log, maxiter = 200, occ_target, hubbard_params = (; t1, t2), mixing_init = 0.2, tol, auto_restart)


    # v2 solver

    # Σ_corr = copy(S.Σ)
    # set!(Σ_corr, 0)
    # mult_add!(Σ_corr, SDE!(copy(Σ_corr), S.G0, S.Π0pp, S.Π0ph, S.L0pp, S.L0ph, S.F0, S.SGΣ, S.SG0pp2, S.SG0ph2; S.mode, include_U² = true, include_Hartree = true), -1)
    # Σ_corr = S.Σ0 - Σ_corr;

    # fdDGAsolver.solve_using_mfRG_v2!(S; filename_log, maxiter = 200, occ_target, hubbard_params = (; t1, t2), mixing_init = 0.2, tol, auto_restart, Σ_corr)


    return S
end

nmax = parse(Int, ARGS[1])
nq   = parse(Int, ARGS[2])
nl_method = parse(Int, ARGS[3])

# filename_log = "/globalscratch/ucl/modl/jmlihm/temp/Wu.mix_bubble.NL$nl_method.nmax$nmax.nq$nq"
filename_log = "/globalscratch/ucl/modl/jmlihm/temp/Wu.mix_bubble.v2.NL$nl_method.nmax$nmax.nq$nq"

S = solve(nmax, nq, nl_method; filename_log);
