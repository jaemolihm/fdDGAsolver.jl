using fdDGAsolver
using MatsubaraFunctions
using HDF5
using MPI
using Test

MPI.Init()
MatsubaraFunctions.DEBUG() = true # enable all checks for testing

println("nthreads = $(Threads.nthreads())")

@testset "fdDGAsolver" begin
    include("test_channel.jl")
    include("test_siam_scPA.jl")
    include("test_siam_fdPA.jl")
    include("test_hubbard.jl")
    include("test_nonlocal_vertex.jl")
    include("test_nonlocal_solver.jl")
    include("test_nonlocal_symmetry.jl")
    include("test_nonlocal_fdPA.jl")

    include("test_nonlocal_2_vertex.jl")
    include("test_nonlocal_2_solver.jl")
    include("test_nonlocal_2_fdPA.jl")
    include("test_fixed_momentum_view.jl")

    include("test_boson_exchange_local.jl")
    include("test_boson_exchange_NL2.jl")

    include("test_flow.jl")
end
