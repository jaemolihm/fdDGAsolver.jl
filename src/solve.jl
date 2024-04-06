
# fixed-point equation
function fixed_point!(
    R::Vector{Q},
    x::Vector{Q},
    S::AbstractSolver{Q}
    ;
    strategy :: Symbol = :fdPA,
    update_Σ :: Bool = true,
)::Nothing where {Q}

    @assert strategy in (:fdPA, :scPA) "Calculation strategy unknown"

    # update solver from input vector
    unflatten!(S, x)

    if update_Σ
        # If Σ is not updated, G and bubbles are already set at the initialization step,
        # so they don't need to be updated here.

        # update G
        Dyson!(S)

        # update bubbles
        bubbles!(S)
    end

    Γpx, F0p, F0a, F0t, Γpp, Γa, Γt, Fp, Fa, Ft = build_K3_cache(S)

    if strategy == :fdPA
        # calculate FL
        BSE_L_K2!(S, pCh)
        BSE_L_K2!(S, aCh)
        BSE_L_K2!(S, tCh)

        BSE_L_K3!(S, Γpp, F0p, pCh)
        BSE_L_K3!(S, Γa,  F0a, aCh)
        BSE_L_K3!(S, Γt,  F0t, tCh)
    end

    # calculate Fbuff
    BSE_K1!(S, pCh)
    BSE_K1!(S, aCh)
    BSE_K1!(S, tCh)

    BSE_K2!(S, pCh)
    BSE_K2!(S, aCh)
    BSE_K2!(S, tCh)

    BSE_K3!(S, Γpx, Fp, F0p, pCh)
    BSE_K3!(S, Γa,  Fa, F0a, aCh)
    BSE_K3!(S, Γt,  Ft, F0t, tCh)

    # update F
    set!(S.F, S.Fbuff)
    reduce!(S.F)

    # update Σ
    if update_Σ

        if strategy === :scPA
            SDE!(S)
        elseif strategy === :fdPA
            # Σ = SDE(ΔΓ, Π, G)
            SDE!(S; include_U² = false, include_Hartree = false)
            #   + SDE(Γ₀, Π, G)
            add!(S.Σ, SDE!(copy(S.Σ), S.G, S.Πpp, S.Πph, S.F0, S.SGΣ, S.SG0pp2, S.SG0ph2; S.mode))
            #   - SDE(Γ₀, Π₀, G₀)
            add!(S.Σ, SDE!(copy(S.Σ), S.G0, S.Π0pp, S.Π0ph, S.F0, S.SGΣ, S.SG0pp2, S.SG0ph2; S.mode) * -1)
            #   + Σ₀
            add!(S.Σ, S.Σ0)

            # # Using K12
            # SDE_using_K12!(S)
            # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G - S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
            # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
        end

        self_energy_sanity_check(S.Σ)

    end

    # calculate residue
    flatten!(S, R)
    R .-= x

    return nothing
end

# run the solver
function solve!(
    S::AbstractSolver,
    ;
    parallel_mode :: Symbol = :serial,
    maxiter  :: Int64 = 100,
    tol      :: Float64 = 1e-4,
    δ        :: Float64 = 0.85,
    mem      :: Int64 = 8,
    verbose  :: Bool = true,
    kwargs_fixed_point...
    )

    verbose && mpi_println("Converging parquet equations.")
    verbose && mpi_println("parallel_mode = $parallel_mode ...")

    S.mode = parallel_mode

    ti = time()
    res = nlsolve((R, x) -> fixed_point!(R, x, S; kwargs_fixed_point...), flatten(S),
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
