using fdDGAsolver
using MatsubaraFunctions
using HDF5
using MPI
using Test

MPI.Init()
MatsubaraFunctions.DEBUG() = true # enable all checks for testing

include("test_channel.jl")