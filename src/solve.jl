# Parquet iteration
function iterate_solver!(S :: AbstractSolver;
    strategy :: Symbol = :fdPA,
    update_Σ :: Bool = true,
    ) ::Nothing

    @assert strategy in (:fdPA, :scPA) "Calculation strategy unknown"

    if update_Σ
        # If Σ is not updated, G and bubbles are already set at the initialization step,
        # so they don't need to be updated here.

        # update G
        Dyson!(S)

        # update bubbles
        bubbles!(S)
    end

    build_K3_cache!(S)

    if strategy == :fdPA
        # calculate FL
        BSE_L_K2!(S, pCh)
        BSE_L_K2!(S, aCh)
        BSE_L_K2!(S, tCh)

        BSE_L_K3!(S, pCh)
        BSE_L_K3!(S, aCh)
        BSE_L_K3!(S, tCh)
    end

    # calculate Fbuff
    BSE_K1!(S, pCh)
    BSE_K1!(S, aCh)
    BSE_K1!(S, tCh)

    BSE_K2!(S, pCh)
    BSE_K2!(S, aCh)
    BSE_K2!(S, tCh)

    BSE_K3!(S, pCh)
    BSE_K3!(S, aCh)
    BSE_K3!(S, tCh)

    # update F
    set!(S.F, S.Fbuff)

    # set!(S.F.γa.K1, 0)
    # set!(S.F.γp.K1, 0)
    # set!(S.F.γt.K1, 0)
    # set!(S.F.γa.K2, 0)
    # set!(S.F.γp.K2, 0)
    # set!(S.F.γt.K2, 0)
    # set!(S.F.γa.K3, 0)
    # set!(S.F.γp.K3, 0)
    # set!(S.F.γt.K3, 0)


    # Symmetrize F (symmetry can be broken during reduction)
    my_symmetrize!(S)

    if update_Σ
        # update self-energy
        SDE!(S; strategy)

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
