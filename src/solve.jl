# Parquet iteration
# `compute_Hartree`: whether to include Hartree term in the self-energy calculation. When
# doing DΓA with Σ0 including the Hartree term, this should be set to `false`.
function iterate_solver!(S :: AbstractSolver;
    strategy :: Symbol = :fdPA,
    update_Σ :: Bool = true,
    compute_Hartree :: Bool = true,
    ) ::Nothing

    @assert strategy in (:fdPA, :scPA, :scPA_new, :fdPA_new, :fdPA_1loop) "Calculation strategy unknown"

    if update_Σ
        # If Σ is not updated, G and bubbles are already set at the initialization step,
        # so they don't need to be updated here.

        # update G
        Dyson!(S)

        # update bubbles
        bubbles!(S)
    end

    build_K3_cache!(S)


    if strategy == :fdPA_new || strategy == :scPA_new
        if strategy == :fdPA_new
            # calculate FL
            BSE_L_K3!(S, pCh)
            BSE_L_K3!(S, aCh)
            BSE_L_K3!(S, tCh)
        end

        BSE_K3!(S, pCh)
        BSE_K3!(S, aCh)
        BSE_K3!(S, tCh)

        # New version
        BSE_K1_new!(S, pCh)
        BSE_K1_new!(S, aCh)
        BSE_K1_new!(S, tCh)

        BSE_K2_new!(S, pCh)
        BSE_K2_new!(S, aCh)
        BSE_K2_new!(S, tCh)

    elseif strategy == :fdPA_1loop
        BSE_K3_1loop!(S, pCh)
        BSE_K3_1loop!(S, aCh)
        BSE_K3_1loop!(S, tCh)

        BSE_K1_1loop!(S, pCh)
        BSE_K1_1loop!(S, aCh)
        BSE_K1_1loop!(S, tCh)

        BSE_K2_1loop!(S, pCh)
        BSE_K2_1loop!(S, aCh)
        BSE_K2_1loop!(S, tCh)
    
    else
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
    end

    # update F
    set!(S.F, S.Fbuff)

    # average_fermionic_momenta!(S.F.γa.K3)
    # average_fermionic_momenta!(S.F.γp.K3)
    # average_fermionic_momenta!(S.F.γt.K3)

    # average_fermionic_momenta!(S.F.γa.K2)
    # average_fermionic_momenta!(S.F.γp.K2)
    # average_fermionic_momenta!(S.F.γt.K2)

    if update_Σ
        # update self-energy
        SDE!(S; strategy, include_Hartree = compute_Hartree)

        # @time for i in 1:10
        #     occ_target = 0.4798972347771976
        #     hubbard_params = (; t1 = 1.0, t2 = -0.3)
        #     μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
        #     set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
        #     Dyson!(S)
        #     bubbles!(S)
        #     SDE!(S; strategy, include_Hartree = compute_Hartree)
        # end

        # self_energy_sanity_check(S.Σ)
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
    update_Σ :: Bool = true,
    callback_chemical_potential_update! = nothing,
    kwargs_solver...
    ) :: Nothing where {Q}

    # update solver from input vector
    if update_Σ
        unflatten!(S, x)

        if callback_chemical_potential_update! !== nothing
            callback_chemical_potential_update!(S)
        end
    else
        unflatten!(S.F, x)
    end

    # Iterate the solver
    iterate_solver!(S; update_Σ, kwargs_solver...)

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
    store_trace :: Bool = true,
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
        store_trace = store_trace,
    )

    verbose && mpi_println("Done. Calculation took $(round(time() - ti, digits = 3)) seconds.")

    history = res.trace.states

    return (; res, history)
end
