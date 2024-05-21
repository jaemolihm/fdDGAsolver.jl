using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using LinearAlgebra
using HDF5

begin
    # System parameters
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)

    data_triqs = load_triqs_data(joinpath(dirname(pathof(fdDGAsolver)), "../data/Wu_point.h5"))

    (; U, T, μ, t1, t2, t3) = data_triqs.params

    occ_target = data_triqs.occ
    hubbard_params = (; t1, t2);
end;

function initialize_solver(nl_method, nmax, nq)
    nG  = 4nmax
    nK1 = 4nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    mG = MatsubaraMesh(T, nG, Fermion)

    mK_G = BrillouinZoneMesh(BrillouinZone(48, k1, k2))
    Gbare = hubbard_bare_Green(mG, mK_G; μ, t1, t2, t3)

    # Set reference Green function and self-energy
    G0 = copy(Gbare)
    Σ0 = copy(Gbare)
    set!(G0, 0)
    set!(Σ0, 0)
    for ν in meshes(G0, Val(1))
        view(G0, ν, :) .= data_triqs.G(value(ν))
        view(Σ0, ν, :) .= data_triqs.Σ(value(ν))
    end

    mK_Γ = BrillouinZoneMesh(BrillouinZone(nq, k1, k2))
    if nl_method == 1
        F0 = NL_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mK_Γ)
        S = NL_ParquetSolver(nK1, nK2, nK3, mK_Γ, copy(Gbare), copy(G0), copy(Σ0), F0)
    elseif nl_method == -2
        F0 = NL2_MBEVertex(fdDGAsolver.asymptotic_to_mbe(data_triqs.Γ), T, nK1, nK2, nK3, mK_Γ)
        S = NL2_ParquetSolver(nK1, nK2, nK3, mK_Γ, copy(Gbare), copy(G0), copy(Σ0), F0, NL2_MBEVertex)
    else
        error("Invalid nl_method $nl_method")
    end
    init_sym_grp!(S)
    return S
end

function initialize_solver_by_interpolation(nl_method, nmax, nq, nmax_in, nq_in; occ_target, hubbard_params)
    Si = initialize_solver(nl_method, nmax_in, nq_in)
    load_solver!(Si, "/globalscratch/ucl/modl/jmlihm/temp/flow.Wu.NL$nl_method.nmax$nmax_in.nq$nq_in.h5")

    S = initialize_solver(nl_method, nmax, nq)
    interpolate_solver!(S, Si; occ_target, hubbard_params)

    # if nl_method == 1
    #     @assert norm(S.F.γp.K1.data) / nq ≈ norm(Si.F.γp.K1.data) / nq_in
    #     @assert norm(S.F.γp.K2.data) / nq ≈ norm(Si.F.γp.K2.data) / nq_in
    #     @assert norm(S.F.γp.K3.data) / nq ≈ norm(Si.F.γp.K3.data) / nq_in
    # elseif nl_method == -2
    #     @assert norm(S.F.γp.K1.data) / nq ≈ norm(Si.F.γp.K1.data) / nq_in
    #     @assert norm(S.F.γp.K2.data) / nq^2 ≈ norm(Si.F.γp.K2.data) / nq_in^2
    #     @assert norm(S.F.γp.K3.data) / nq ≈ norm(Si.F.γp.K3.data) / nq_in
    # end

    return S
end


begin
    # nl_method = -2
    # nmax, nq = 6, 3
    # nmax_in, nq_in = 4, 3
    nl_method = parse(Int, ARGS[1])
    nmax = parse(Int, ARGS[2])
    nq = parse(Int, ARGS[3])
    nmax_in = parse(Int, ARGS[4])
    nq_in = parse(Int, ARGS[5])
    
    # S0 = initialize_solver(nl_method)

    S0 = initialize_solver_by_interpolation(nl_method, nmax, nq, nmax_in, nq_in; occ_target, hubbard_params)

    # res = fdDGAsolver.solve!(S0; strategy = :fdPA, maxiter=50, mem=50, tol=1e-5);
    for i in 1:10
        println("Outer iteration $i")
        flush(stdout)
        flush(stderr)

        res = fdDGAsolver.solve_using_mfRG_without_mixing!(S0; strategy = (nl_method > 0 ? :fdPA : :fdPA_new), tol=1e-5, update_Σ = true, occ_target, hubbard_params, memory=100, maxiter=50);
        unflatten!(S0, res.zero);

        filename = "/globalscratch/ucl/modl/jmlihm/temp/flow.Wu.NL$nl_method.nmax$nmax.nq$nq.h5"
        h5open(filename, "w") do f
            save!(f, "S", S0)
        end
        println("Saved nl_method = $nl_method, nmax = $nmax, nq = $nq")

        if res.f_converged
            break
        end
    end
end;
