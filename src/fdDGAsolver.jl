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

    include("types.jl")
    include("symmetries.jl")
    # include("sum_me.jl")
    include("channel.jl")
    include("refvertex.jl")
    include("vertex.jl")

    include("models/siam.jl")
    include("models/hubbard.jl")

    include("ParquetSolver.jl")
    include("bubble.jl")
    include("SDE.jl")
    include("BSE.jl")
    include("solve.jl")

    export
        pCh, tCh, aCh,
        pSp, xSp, dSp,
        numK1, numK2, numK3,
        Channel,
        RefVertex,
        Vertex,
        ParquetSolver

end
