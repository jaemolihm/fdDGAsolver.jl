# Parquet iteration
function iterate_solver!(S :: AbstractSolver;
    strategy :: Symbol = :fdPA,
    update_Σ :: Bool = true,
    ) ::Nothing

    @assert strategy in (:fdPA, :scPA) "Calculation strategy unknown"

    zero_out_points_outside_box!(S)

    if update_Σ
        # If Σ is not updated, G and bubbles are already set at the initialization step,
        # so they don't need to be updated here.

        # update G
        Dyson!(S)

        # update bubbles
        bubbles!(S)
    end


    (; Γpx, F0p, F0a, F0t, Γpp, Γa, Γt, Fp, Fa, Ft) = build_K3_cache(S)


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

    zero_out_points_outside_box!(S)

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
            mult_add!(S.Σ, SDE!(copy(S.Σ), S.G0, S.Π0pp, S.Π0ph, S.F0, S.SGΣ, S.SG0pp2, S.SG0ph2; S.mode), -1)
            #   + Σ₀
            add!(S.Σ, S.Σ0)

            # # Using K12
            # SDE_using_K12!(S)
            # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G - S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
            # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
        end

        self_energy_sanity_check(S.Σ)

    end

    return nothing

end

# fixed-point equation
function fixed_point!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver{Q}
    ;
    filename_log :: Union{Nothing, String} = nothing,
    kwargs_solver...
    ) :: Nothing where {Q}

    # update solver from input vector
    unflatten!(S, x)

    # Iterate the solver
    iterate_solver!(S; kwargs_solver...)

    if filename_log !== nothing
        if mpi_ismain()
            h5open(filename_log, "w") do f
                save!(f, "S", S)
            end
        end
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
    parallel_mode :: Union{Nothing, Symbol} = nothing,
    maxiter  :: Int64 = 100,
    tol      :: Float64 = 1e-4,
    δ        :: Float64 = 0.85,
    mem      :: Int64 = 8,
    verbose  :: Bool = true,
    kwargs_solver...  # passed to the iterate_solver function
    )

    if parallel_mode !== nothing
        S.mode = parallel_mode
    end

    verbose && mpi_println("Converging parquet equations.")
    verbose && mpi_println("parallelization mode : $(S.mode)")

    ti = time()
    res = nlsolve((R, x) -> fixed_point!(R, x, S; kwargs_solver...), flatten(S),
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
