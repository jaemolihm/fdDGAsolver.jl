using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using StaticArrays

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
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

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
    a = fdDGAsolver.mfRGLinearMap(S);
    x = flatten(S.F);
    @time a * x;
    @time a * x;
    @time a * x;

    mpi_barrier()

    mpi_ismain() && println(" === solve_using_mfRG! ===")
    F0 = NL2_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mK_Γ)
    S = NL2_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, copy(G0), copy(Σ0), F0; mode = :hybrid)
    init_sym_grp!(S)
    occ_target = compute_occupation(S.G)
    @time fdDGAsolver.solve_using_mfRG!(S; filename_log = nothing, maxiter = 1, occ_target, hubbard_params = (; t1, t2), mixing_init = 0.2, tol=100.0);

    return nothing
end

run_benchmark(4, 8);
