using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using LinearAlgebra
using HDF5
using MatsubaraFunctions: mesh_index
using fdDGAsolver: numP_Γ, k0, kSW

function run_benchmark(nmax, nq)
    # System parameters
    T = 0.2
    U = 5.6
    μ = 2.1800201007694464 - U/2
    t1 = 1.0
    t2 = -0.3

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    data_triqs = load_triqs_data("/home/ucl/modl/jmlihm/MFjl/fdDGAsolver.jl/data/Wu_point.h5")

    nG  = 4nmax
    nK1 = 4nmax
    nK2 = (nmax + 1, nmax)
    nK3 = (nmax + 1, nmax)

    mK_G = BrillouinZoneMesh(BrillouinZone(48, k1, k2))
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

    F0 = NL2_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mK_Γ)
    S = NL2_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, copy(G0), copy(Σ0), F0; mode = :hybrid)
    init_sym_grp!(S)

    if mpi_ismain()
        @info "nmax = $nmax, nq = $nq"
        @info "mpi_size = $(mpi_size())"
        @info "nthreads = $(Threads.nthreads())"
    end

    mpi_ismain() && println(" === iterate_solver! ===")
    @time iterate_solver!(S; update_Σ = false, strategy = :fdPA)
    @time iterate_solver!(S; update_Σ = false, strategy = :fdPA)
    @time iterate_solver!(S; update_Σ = false, strategy = :fdPA)

    mpi_barrier()

    mpi_ismain() && println(" === mfRGLinearMap ===")
    a = mfRGLinearMap(S)
    x = flatten(S.F)
    @time a * x
    @time a * x
    @time a * x
end

run_benchmark(6, 6)
