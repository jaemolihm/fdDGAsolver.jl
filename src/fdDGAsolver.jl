__precompile__(false)
module fdDGAsolver

    using PrecompileTools

    @recompile_invalidations begin
        using MPI
        using MatsubaraFunctions
        using Polyester
        using HDF5
    end

    include("types.jl")
    include("channel_frequency.jl")
    include("channel_spin.jl")
    include("vertex.jl")

end
