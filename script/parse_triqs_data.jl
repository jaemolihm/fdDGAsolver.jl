using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using HDF5

include("/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/script/plot_vertex.jl")

begin
    # System parameters : high temperature
    T = 0.5
    t1 = -0.25
    t2 = 0.0
    t3 = 0.0
    μ = 0.0
    # for U in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.04, 2.06, 2.08, 2.1, 2.12]
    for U in [0.75, 1.0, 1.25, 1.5, 1.75]
    # for U in [2.0875, 2.0885, 2.089, 2.0895, 2.8, 2.825, 2.85, 2.875, 2.9]
        # Load impurity vertex
        prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta2.0_t-0.25_U$(U)_mu$(U/2)_numc1_numk256"
        filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/high_temperature_U$U.h5"
        data_triqs = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1, t2, t3), half_filling = true, symmetrize = true, filename_output)
        # plot_vertex_K1(Γ)
        # plot_vertex_K2(Γ; vmax = 0.01)
        # plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)
    end
end;


begin
    # System parameters : Wu point
    T = 0.2
    U = 5.6
    μ = 2.1800201007694464 - U/2
    t1 = 1.0
    t2 = -0.3
    t3 = 0.0

    # Load impurity vertex
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta5.0_t-1.0_U5.6_mu2.1800201007694464_numc1_numk255"
    filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point.h5"

    data_triqs = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1, t2, t3), half_filling = false, symmetrize = true, filename_output)
    # plot_vertex_K1(Γ)
    # plot_vertex_K2(Γ; vmax = 0.01)
    # plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)
end;


begin
    # System parameters : Wu point - update May 3 2024
    T = 0.2
    t1 = 1.0
    t2 = -0.3
    t3 = 0.0
    # for (U, μ) in [(5.0, 1.8622674424442904),
    #                (5.2, 1.9690673134296466),
    #                (5.4, 2.0734335491694544),
    #                (5.6, 2.178910712243582),
    #                (5.8, 2.2840101729704414),
    #                (6.0, 2.3877368985791043),
                #    (6.2, 2.490307386373797),
                #    (5.5, 2.125315907991539),
                # (5.55, 2.1537018006864894),
    # ]
    for (U, μ) in [(5.55, 2.1537018006864894)]

        # Load impurity vertex
        prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta5.0_t-1.0_U$(U)_mu$(μ)_numc1_numk254"
        filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point_U$U.h5"

        μ = μ - U/2

        data_triqs = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1, t2, t3), half_filling = false, symmetrize = true, filename_output)
        # plot_vertex_K1(data_triqs.Γ)
        # plot_vertex_K2(Γ; vmax = 0.01)
        # plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)
    end
end;



begin
    # System parameters : weak couplig (Eckhardt TUPS) points
    T = 0.2
    μ = 0.0
    t1 = -1.0
    t2 = 0.0
    t3 = 0.0
    for U in [2.0, 4.0]
        prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta5.0_t-1.0_U$(U)_mu$(U/2)_numc1_numk254"
        (; G0, G, Σ, Γ, occ) = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1), half_filling = true)
        filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/weak_coupling_U$U.h5"

        data_triqs = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1), half_filling = true, symmetrize = true, filename_output)
        # plot_vertex_K1(Γ)
        # plot_vertex_K2(Γ; vmax = 0.01)
        # plot_vertex_core(Γ; vmax = 0.1, Ω = 2π*T)
    end
end;

begin
    # System parameters : Krien point
    T = 0.15
    U = 8.0
    t1 = 1.0
    t2 = -0.2
    t3 = 0.1

    # Load impurity vertex
    # prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta6.6667_t-1.0_U8.0_mu3.8217923274829233_numc1_numk254"
    # filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Krien_point.h5"
    # μ = 3.8217923274829233 - U/2

    # Update Aug 5 2024 (larger frequency box)
    # prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta6.6667_t-1.0_U8.0_mu3.8100430368818334_numc1_numk254"
    # filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Krien_point_new.h5"
    # μ = 3.8100430368818334 - U/2

    # Update Aug 20 2024 (different doping)
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta6.6667_t-1.0_U8.0_mu1.8146053242102085_numc1_numk254"
    filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Krien_point_new_doping_0.15.h5"
    μ = 1.8146053242102085 - U/2


    data_triqs = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1, t2, t3), half_filling = false, symmetrize = true, filename_output)
    plot_vertex_K1(data_triqs.Γ)
    plot_vertex_K2(data_triqs.Γ; vmax = 10.)
    plot_vertex_core(data_triqs.Γ; vmax = 20., Ω = 2π*T*0)
end;


begin
    # System parameters : Wu point - update Nov 18 2024
    T = 0.2
    t1 = 1.0
    t2 = -0.3
    t3 = 0.0
    U = 5.6
    μ = 2.178910712243582

    # Load impurity vertex
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/new_2024.11.18/beta5.0_t-1.0_U$(U)_mu$(μ)_numc1_numk254"
    filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point_new_2024.11.18_U$U.h5"

    μ = μ - U/2

    data_triqs = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1, t2, t3), half_filling = false, symmetrize = true, filename_output);

    plot_vertex_K2(data_triqs.Γ; vmax = 1);
    plot_vertex_core(data_triqs.Γ; vmax = 0.1);
end;



begin
    # System parameters : weak coupling point from Schaefer et al, PRX (2021)
    U = 2.0
    μ = 0.0
    t1 = 1.0
    t2 = 0.0
    t3 = 0.0
    
    # Load impurity vertex
    T = 0.1
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta10.0_t-1.0_U2.0_mu1.0000303034649165_numc1_numk254"

    T = 0.067
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta14.925_t-1.0_U2.0_mu0.9999352513262938_numc1_numk254"

    T = 0.065; beta = 15.3846
    T = 0.063; beta = 15.873
    T = 0.061; beta = 16.3934
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta$(beta)_t-1.0_U2.0_mu1.0_numc1_numk254"

    filename_output = "/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Schaefer_T$T.h5"

    data_triqs = fdDGAsolver.parse_triqs_data(prefix, T, U; params = (; μ, t1, t2, t3), half_filling = true, symmetrize = true, filename_output)
    plot_vertex_K1(data_triqs.Γ)
    plot_vertex_K2(data_triqs.Γ; vmax = 0.1)
    plot_vertex_core(data_triqs.Γ; vmax = 0.1, Ω = 2π*T*0)
end;
