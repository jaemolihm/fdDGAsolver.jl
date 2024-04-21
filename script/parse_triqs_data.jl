using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using HDF5


begin
    # System parameters : high temperature
    T = 0.5
    U = 2.089
    μ = 0.0
    t1 = -0.25

    # Load impurity vertex
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta2.0_t-0.25_U2.089_mu1.0445_numc1_numk256"
    filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/high_temperature_U2.089.h5"
    data_triqs = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1), half_filling = true, symmetrize = true, filename_output)
    # plot_vertex_K1(Γ)
    # plot_vertex_K2(Γ; vmax = 0.01)
    # plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)
end;


begin
    # System parameters : Wu point
    T = 0.2
    U = 5.6
    μ = 2.1800201007694464 - U/2
    t1 = 1.0
    t2 = -0.3

    # Load impurity vertex
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta5.0_t-1.0_U5.6_mu2.1800201007694464_numc1_numk255"
    filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point.h5"

    data_triqs = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1, t2), half_filling = false, symmetrize = true, filename_output)
    # plot_vertex_K1(Γ)
    # plot_vertex_K2(Γ; vmax = 0.01)
    # plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)
end;
