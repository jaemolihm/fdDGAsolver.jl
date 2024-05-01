__precompile__(false)
module fdDGAsolver

    using PrecompileTools

    @recompile_invalidations begin
        using MPI
        MPI.Init()
        using MatsubaraFunctions
        using Polyester
        using HDF5
        using NLsolve
        using LinearAlgebra
        using StaticArrays
        using FFTW
        using Roots
        using Krylov
        using LinearMaps
    end

    include("types.jl")
    include("convention.jl")
    include("symmetries.jl")
    include("channel.jl")
    include("refvertex.jl")
    include("vertex.jl")

    include("models/siam.jl")
    include("models/hubbard.jl")

    include("ParquetSolver.jl")
    include("bubble.jl")
    include("dyson.jl")
    include("SDE.jl")
    include("BSE_templates.jl")
    include("BSE.jl")
    include("build_K3_cache.jl")
    include("solve.jl")
    include("utility/load_triqs.jl")

    include("nonlocal/swave.jl")
    include("nonlocal/symmetries.jl")
    include("nonlocal/channel.jl")
    include("nonlocal/vertex.jl")
    include("nonlocal/bubble.jl")
    include("nonlocal/ParquetSolver.jl")

    include("nonlocal_2/symmetries.jl")
    include("nonlocal_2/channel.jl")
    include("nonlocal_2/vertex.jl")
    include("nonlocal_2/ParquetSolver.jl")

    include("nonlocal/SDE.jl")
    include("nonlocal/BSE.jl")
    include("nonlocal/build_K3_cache.jl")

    include("nonlocal_2/bubble.jl")
    include("nonlocal_2/BSE.jl")
    include("nonlocal_2/SDE.jl")
    include("nonlocal_2/build_K3_cache.jl")

    include("zero_out.jl")

    include("fixed_momentum_view.jl")

    include("mfRG.jl")

    include("nonlocal_2/mbe.jl")

    include("boson_exchange.jl")


    @compile_workload begin
        MPI.Init()

        T = 0.2
        U = 2.0
        μ = 0.0
        t1 = 1.0

        nmax = 2
        nG  = 12nmax
        nK1 = 8nmax
        nK2 = (nmax, nmax)
        nK3 = (nmax, nmax)

        k1 = 2pi * SVector(1., 0.)
        k2 = 2pi * SVector(0., 1.)
        mK_G = BrillouinZoneMesh(BrillouinZone(2, k1, k2))
        mK_Γ = BrillouinZoneMesh(BrillouinZone(2, k1, k2))

        S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e = 0., T, U, Δ = 3., D = 10., mode = :hybrid)
        init_sym_grp!(S)
        solve!(S; strategy = :fdPA, maxiter = 1, verbose = false)
        unflatten!(S, flatten(S))

        S = parquet_solver_hubbard_parquet_approximation(nG, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1, mode = :hybrid)
        init_sym_grp!(S)
        solve!(S; strategy = :fdPA, maxiter = 1, verbose = false)
        unflatten!(S, flatten(S))

        S = parquet_solver_hubbard_parquet_approximation_NL2(nG, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1, mode = :hybrid)
        init_sym_grp!(S)
        solve!(S; strategy = :fdPA, maxiter = 1, verbose = false)
        unflatten!(S, flatten(S))

        # ----------------------------------------------------------------------------
        # Parquet DΓA

        # System parameters
        T = 0.2
        U = 5.6
        μ = 2.1800201007694464 - U/2
        t1 = 1.0
        t2 = -0.3

        k1 = 2pi * SVector(1., 0.)
        k2 = 2pi * SVector(0., 1.)

        data_triqs = load_triqs_data(joinpath(dirname(@__FILE__), "../data/Wu_point.h5"))

        nmax = 1
        nG  = 4nmax
        nK1 = 4nmax
        nK2 = (nmax, nmax)
        nK3 = (nmax, nmax)

        nq = 2
        mK_G = BrillouinZoneMesh(BrillouinZone(4, k1, k2))
        mK_Γ = BrillouinZoneMesh(BrillouinZone(nq, k1, k2))

        mG = MatsubaraMesh(T, nG, Fermion)
        Gbare = hubbard_bare_Green(mG, mK_G; μ, t1, t2)

        # Set reference Green function and self-energy
        G0 = copy(Gbare)
        Σ0 = copy(Gbare)
        set!(G0, 0)
        set!(Σ0, 0)
        for ν in meshes(G0, Val(1))
            view(G0, ν, :) .= data_triqs.G(value(ν))
            view(Σ0, ν, :) .= data_triqs.Σ(value(ν))
        end
        occ_target = data_triqs.occ

        F0 = NL2_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mK_Γ)
        S = NL2_ParquetSolver(nK1, nK2, nK3, mK_Γ, copy(Gbare), copy(G0), copy(Σ0), F0; mode = :hybrid)
        init_sym_grp!(S)
        solve_using_mfRG!(S; maxiter = 1, occ_target, hubbard_params = (; t1, t2), tol = 1e3, verbose=false);

        Σ_corr = copy(S.Σ0)
        mult_add!(Σ_corr, SDE!(copy(Σ_corr), S.F0, S.G0, S.Π0pp, S.Π0ph, S; include_U² = true, include_Hartree = true), -1)
        solve_using_mfRG_v2!(S; maxiter = 1, occ_target, hubbard_params = (; t1, t2), tol = 1e3, verbose=false, Σ_corr);
    end

    export
        pCh, tCh, aCh,
        pSp, xSp, dSp,
        νInf,
        Channel,
        RefVertex,
        Vertex,
        AbstractSolver,
        NL_Channel, NL_Vertex,
        NL2_Channel, NL2_Vertex,
        compute_occupation,
        init_sym_grp!,
        ParquetSolver,
        NL_ParquetSolver,
        NL2_ParquetSolver,
        parquet_solver_siam_parquet_approximation,
        parquet_solver_hubbard_parquet_approximation,
        parquet_solver_hubbard_parquet_approximation_NL2,
        SDE!, BSE_K1!, BSE_K2!, BSE_K3!, BSE_L_K2!, BSE_L_K3!, build_K3_cache, iterate_solver!,
        get_P_mesh, Dyson!, bubbles!,
        load_triqs_data, compute_hubbard_chemical_potential,
        MBEVertex, NL2_MBEVertex

end
