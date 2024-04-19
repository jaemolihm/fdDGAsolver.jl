using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using LinearAlgebra
using HDF5
using MatsubaraFunctions: mesh_index
using fdDGAsolver: numP_Γ, k0, kSW
using fdDGAsolver: sK1pp, sK2pp1, sK2pp2, sK3pp1, sK3pp2, sK3pp3, my_SymmetryGroup, sK1ph, sK2ph1, sK2ph2, sK3ph1, sK3ph2, sK3ph3, mfRGLinearMap

function run_benchmark(nmax, nq)
    # System parameters
    T = 0.2
    U = 5.6
    μ = 2.1800201007694464 - U/2
    t1 = 1.0
    t2 = -0.3

    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    # Load impurity vertex
    prefix = "/home/ucl/modl/jmlihm/MFjl/data/beta5.0_t-1.0_U5.6_mu2.1800201007694464_numc1_numk255"
    data_triqs = load_vertex_from_triqs(prefix, T, U; half_filling = false)

    # Symmetrize impurity vertex
    Γ = data_triqs.Γ

    # particle-particle channel
    SGpp1 = my_SymmetryGroup([Symmetry{1}(sK1pp)], Γ.γp.K1)
    SGpp2 = my_SymmetryGroup([Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)], Γ.γp.K2)
    SGpp3 = my_SymmetryGroup([Symmetry{3}(sK3pp1), Symmetry{3}(sK3pp2), Symmetry{3}(sK3pp3)], Γ.F0.Fp_p)

    # particle-hole channels
    SGph1 = my_SymmetryGroup([Symmetry{1}(sK1ph)], Γ.γt.K1)
    SGph2 = my_SymmetryGroup([Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)], Γ.γt.K2)
    SGph3 = my_SymmetryGroup([Symmetry{3}(sK3ph1), Symmetry{3}(sK3ph2), Symmetry{3}(sK3ph3)], Γ.F0.Ft_p)

    SGpp1(Γ.γp.K1)
    SGpp2(Γ.γp.K2)
    SGph1(Γ.γt.K1)
    SGph1(Γ.γa.K1)
    SGph2(Γ.γt.K2)
    SGph2(Γ.γa.K2)
    SGpp3(Γ.F0.Fp_p)
    SGpp3(Γ.F0.Fp_x)
    SGph3(Γ.F0.Ft_p)
    SGph3(Γ.F0.Ft_x)

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
