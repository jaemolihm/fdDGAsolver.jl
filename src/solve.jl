
# fixed-point equation
function fixed_point!(
    R::Vector{Q},
    x::Vector{Q},
    S::ParquetSolver
    ;
    strategy :: Symbol = :fdPA
)::Nothing where {Q}

    @assert strategy in (:fdPA, :scPA) "Calculation strategy unknown"

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

    Threads.@threads for i in CartesianIndices(Γpx.data)
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

    Threads.@threads for i in CartesianIndices(Γpp.data)
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

    if strategy == :fdPA
        # calculate FL
        BSE_L_K2!(S, pCh)
        BSE_L_K2!(S, tCh)
        BSE_L_K2!(S, aCh)

        BSE_L_K3!(S, Γpp, F0p, pCh)
        BSE_L_K3!(S, Γt, Γa, F0t, F0a, tCh)
        BSE_L_K3!(S, Γa, F0a, aCh)
    end

    # calculate Fbuff
    # println("BSE_K1")
    BSE_K1!(S, pCh)
    BSE_K1!(S, tCh)
    BSE_K1!(S, aCh)

    # println("BSE_K2")
    BSE_K2!(S, pCh)
    BSE_K2!(S, tCh)
    BSE_K2!(S, aCh)

    # println("BSE_K3")
    BSE_K3!(S, Γpx, Fp, F0p, pCh)
    BSE_K3!(S, Γt, Γa, Ft, Fa, F0t, F0a, tCh)
    BSE_K3!(S, Γa, Fa, F0a, aCh)

    # update F
    set!(S.F, S.Fbuff)
    reduce!(S.F)

    # update Σ
    if strategy === :scPA
        SDE!(S)
    elseif strategy === :fdPA
        # Σ = SDE(ΔΓ, Π, G)
        SDE!(S; include_U² = false, include_Hartree = false)
        #   + SDE(Γ₀, Π, G)
        add!(S.Σ, SDE!(copy(S.Σ), S.G, S.Πpp, S.Πph, S.F0, S.SGΣ, S.SGpp[2], S.SGph[2]; S.mode))
        #   - SDE(Γ₀, Π₀, G₀)
        add!(S.Σ, SDE!(copy(S.Σ), S.G0, S.Π0pp, S.Π0ph, S.F0, S.SGΣ, S.SGpp[2], S.SGph[2]; S.mode) * -1)
        #   + Σ₀
        add!(S.Σ, S.Σ0)

        # # Using K12
        # SDE_using_K12!(S)
        # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G - S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
        # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
    end

    self_energy_sanity_check(S.Σ)

    # calculate residue
    flatten!(S, R)
    R .-= x

    return nothing
end

# run the solver
function solve!(
    S::ParquetSolver
    ;
    parallel_mode :: Symbol = :serial,
    strategy::Symbol = :fdPA,
    maxiter::Int64=100,
    tol::Float64=1e-4,
    δ::Float64=0.85,
    mem::Int64=8,
    verbose :: Bool = true,
)

    verbose && mpi_println("Converging parquet equations.")
    verbose && mpi_println("parallel_mode = $parallel_mode ...")

    S.mode = parallel_mode

    ti = time()
    res = nlsolve((R, x) -> fixed_point!(R, x, S; strategy), flatten(S),
        method=:anderson,
        iterations=maxiter,
        ftol=tol,
        beta=δ,
        m=mem,
        show_trace = mpi_ismain() && verbose,
    )

    verbose && mpi_println("Done. Calculation took $(round(time() - ti, digits = 3)) seconds.")

    return res
end

# save solver to HDF5
function MatsubaraFunctions.save!(
    f::HDF5.File,
    label::String,
    S::ParquetSolver
)::Nothing

    MatsubaraFunctions.save!(f, "G0", S.G0)
    MatsubaraFunctions.save!(f, "Σ0", S.Σ0)
    MatsubaraFunctions.save!(f, "F0", S.F0)

    MatsubaraFunctions.save!(f, "G", S.G)
    MatsubaraFunctions.save!(f, "Σ", S.Σ)
    MatsubaraFunctions.save!(f, "F", S.F)

    return nothing
end

export
    parquet_solver_siam_parquet_approximation
