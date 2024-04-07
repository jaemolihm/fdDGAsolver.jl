__precompile__(true)
module fdDGAsolver

    using PrecompileTools

    @recompile_invalidations begin
        using MPI
        MPI.Init()
        using MatsubaraFunctions
        using Polyester
        using HDF5
        using NLsolve
        using StaticArrays
    end

    include("matsubarafunctions_piracy.jl")
    include("box_eval.jl")

    include("types.jl")
    include("convention.jl")
    include("symmetries.jl")
    # include("sum_me.jl")
    include("channel.jl")
    include("refvertex.jl")
    include("vertex.jl")

    include("models/siam.jl")
    include("models/hubbard.jl")

    include("ParquetSolver.jl")
    include("bubble.jl")
    include("dyson.jl")
    include("SDE.jl")
    include("BSE.jl")
    include("build_K3_cache.jl")
    include("solve.jl")

    include("nonlocal/swave.jl")
    include("nonlocal/symmetries.jl")
    include("nonlocal/channel.jl")
    include("nonlocal/vertex.jl")
    include("nonlocal/ParquetSolver.jl")

    include("nonlocal_2/symmetries.jl")
    include("nonlocal_2/channel.jl")
    include("nonlocal_2/vertex.jl")
    include("nonlocal_2/ParquetSolver.jl")

    include("nonlocal/SDE.jl")
    include("nonlocal/BSE.jl")
    include("nonlocal/build_K3_cache.jl")

    include("nonlocal_2/BSE.jl")


    @compile_workload begin
        MPI.Init()

        T = 0.2
        U = 2.0
        μ = 0.0
        t1 = 1.0

        nmax = 2
        nG  = 12nmax
        nΣ  = 12nmax
        nK1 = 8nmax
        nK2 = (2nmax, nmax)
        nK3 = (2nmax, nmax)

        k1 = 2pi * SVector(1., 0.)
        k2 = 2pi * SVector(0., 1.)
        mK_G = BrillouinZoneMesh(BrillouinZone(4, k1, k2))
        mK_Γ = BrillouinZoneMesh(BrillouinZone(2, k1, k2))

        S = parquet_solver_hubbard_parquet_approximation(nG, nΣ, nK1, nK2, nK3, mK_G, mK_Γ; T, U, μ, t1, mode = :hybrid)
        iterate_solver!(S; strategy = :scPA)
    end

    export
        pCh, tCh, aCh,
        pSp, xSp, dSp,
        νInf,
        Channel,
        RefVertex,
        Vertex,
        NL_Channel, NL_Vertex,
        NL2_Channel, NL2_Vertex,
        compute_occupation,
        ParquetSolver,
        NL_ParquetSolver,
        parquet_solver_siam_parquet_approximation,
        parquet_solver_hubbard_parquet_approximation,
        parquet_solver_hubbard_parquet_approximation_NL2

end
