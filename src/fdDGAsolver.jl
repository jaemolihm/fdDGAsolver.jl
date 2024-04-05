__precompile__(false)
module fdDGAsolver

    using PrecompileTools

    @recompile_invalidations begin
        using MPI
        using MatsubaraFunctions
        using Polyester
        using HDF5
        using NLsolve
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

    include("nonlocal/symmetries.jl")
    include("nonlocal/channel.jl")
    include("nonlocal/vertex.jl")
    include("nonlocal/ParquetSolver.jl")
    include("nonlocal/BSE.jl")

    include("solve.jl")

    export
        pCh, tCh, aCh,
        pSp, xSp, dSp,
        νInf,
        Channel,
        RefVertex,
        Vertex,
        NL_Channel, NL_Vertex,
        ParquetSolver,
        NL_ParquetSolver,
        parquet_solver_hubbard_parquet_approximation

end
