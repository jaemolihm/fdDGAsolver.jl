using fdDGAsolver
using MatsubaraFunctions
using HDF5
using MPI
using Test

MPI.Init()
MatsubaraFunctions.DEBUG() = true # enable all checks for testing

@testset "fdDGAsolver" begin
    include("test_channel.jl")
    include("test_siam_scPA.jl")
    include("test_siam_fdPA.jl")
    include("test_hubbard.jl")
    include("test_nonlocal_vertex.jl")
    include("test_nonlocal_solver.jl")
    include("test_nonlocal_symmetry.jl")
end
