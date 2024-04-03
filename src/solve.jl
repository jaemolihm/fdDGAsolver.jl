
# fixed-point equation
function fixed_point!(
    R::Vector{Q},
    x::Vector{Q},
    S::ParquetSolver
    ;
    strat::Symbol=:fdPA
)::Nothing where {Q}

    @assert strat in (:fdPA, :scPA) "Calculation strategy unknown"

    # update solver from input vector
    unflatten!(S, x)

    # update G
    Dyson!(S)

    # update bubbles
    bubbles!(S)

    # build vertices
    # Γpx : Target   , p-channel irreducible, spin cross
    # F0p : Reference, p-channel full       , spin cross
    # F0t : Reference, t-channel full       , spin parallel
    # F0a : Reference, a-channel full       , spin parallel
    g = MatsubaraMesh(temperature(S), 2 * numK1(S), Fermion)
    Γpx = MeshFunction(meshes(S.F.γp.K3, 1), g, meshes(S.F.γp.K3, 2); data_t=Q)
    F0p = copy(Γpx)
    F0t = copy(Γpx)
    F0a = copy(Γpx)

    @time Threads.@threads for i in CartesianIndices(Γpx.data)
        Ω = value(meshes(Γpx, 1)[i.I[1]])
        ν = value(meshes(Γpx, 2)[i.I[2]])
        νp = value(meshes(Γpx, 3)[i.I[3]])
        Γpx[i] = S.F(Ω, ν, νp, pCh, xSp; F0=false, γp=false)
        F0p[i] = S.F0(Ω, ν, νp, pCh, xSp)
        F0t[i] = S.F0(Ω, ν, νp, tCh, pSp)
        F0a[i] = S.F0(Ω, ν, νp, aCh, pSp)
    end

    # Γpp : Target, p-channel irreducible, spin parallel
    # Γt  : Target, t-channel irreducible, spin parallel
    # Γa  : Target, a-channel irreducible, spin parallel
    # Fp  : Target, a-channel full,        spin parallel
    # Ft  : Target, a-channel full,        spin parallel
    # Fa  : Target, a-channel full,        spin parallel
    Γpp = MeshFunction(meshes(S.F.γp.K3, 1), meshes(S.F.γp.K3, 2), g; data_t=Q)
    Γt = deepcopy(Γpp)
    Γa = deepcopy(Γpp)
    Fp = deepcopy(Γpp)
    Ft = deepcopy(Γpp)
    Fa = deepcopy(Γpp)

    @time Threads.@threads for i in CartesianIndices(Γpp.data)
        Ω = value(meshes(Γpp, 1)[i.I[1]])
        ν = value(meshes(Γpp, 2)[i.I[2]])
        νp = value(meshes(Γpp, 3)[i.I[3]])
        Γpp[i] = S.F(Ω, ν, νp, pCh, pSp; F0=false, γp=false)
        Γt[i] = S.F(Ω, ν, νp, tCh, pSp; F0=false, γt=false)
        Γa[i] = S.F(Ω, ν, νp, aCh, pSp; F0=false, γa=false)
        Fp[i] = S.F(Ω, ν, νp, pCh, pSp)
        Ft[i] = S.F(Ω, ν, νp, tCh, pSp)
        Fa[i] = S.F(Ω, ν, νp, aCh, pSp)
    end

    if strat == :fdPA
        # calculate FL
        BSE_L_K2!(S, pCh)
        BSE_L_K2!(S, tCh)
        BSE_L_K2!(S, aCh)

        BSE_L_K3!(S, Γpp, F0p, pCh)
        BSE_L_K3!(S, Γt, Γa, F0t, F0a, tCh)
        BSE_L_K3!(S, Γa, F0a, aCh)
    end

    # calculate Fbuff
    println("BSE_K1")
    @time BSE_K1!(S, pCh)
    @time BSE_K1!(S, tCh)
    @time BSE_K1!(S, aCh)

    println("BSE_K2")
    @time BSE_K2!(S, pCh)
    @time BSE_K2!(S, tCh)
    @time BSE_K2!(S, aCh)

    println("BSE_K3")
    @time BSE_K3!(S, Γpx, Fp, F0p, pCh)
    @time BSE_K3!(S, Γt, Γa, Ft, Fa, F0t, F0a, tCh)
    @time BSE_K3!(S, Γa, Fa, F0a, aCh)

    # update F
    set!(S.F, S.Fbuff)
    @time reduce!(S.F)

    # update Σ
    # SDE!(S)
    @time SDE_channel!(S)
    @time Σ_U² = SDE_U2(S)
    add!(S.Σ, Σ_U²)

    # calculate residue
    flatten!(S, R)
    R .-= x

    return nothing
end

# run the solver
function solve!(
    S::ParquetSolver
    ;
    strat::Symbol=:fdPA,
    maxiter::Int64=100,
    tol::Float64=1e-4,
    δ::Float64=0.85,
    mem::Int64=8
)

    mpi_println("Converging parquet equations ...")

    ti = time()
    res = nlsolve((R, x) -> fixed_point!(R, x, S; strat), flatten(S),
        method=:anderson,
        iterations=maxiter,
        ftol=tol,
        beta=δ,
        m=mem,
        show_trace=true)
        # mpi_ismain())

    # mpi_println("Done. Calculation took $(round(time() - ti, digits = 3)) seconds.")

    return res
end

# save solver to HDF5
function MatsubaraFunctions.save!(
    f::HDF5.File,
    label::String,
    S::ParquetSolver
)::Nothing

    attributes(f)["V"] = S.V
    attributes(f)["D"] = S.D
    attributes(f)["νInf"] = index(S.νInf)

    MatsubaraFunctions.save!(f, "G0", S.G0)
    MatsubaraFunctions.save!(f, "Σ0", S.Σ0)
    MatsubaraFunctions.save!(f, "F0", S.F0)

    MatsubaraFunctions.save!(f, "G", S.G)
    MatsubaraFunctions.save!(f, "Σ", S.Σ)
    MatsubaraFunctions.save!(f, "F", S.F)

    return nothing
end