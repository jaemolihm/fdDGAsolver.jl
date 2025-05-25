function unique_filename_log(filename_log :: Union{String, Nothing})
    # If filename_log ends with .h5, return it.
    # Otherwise, append ".iter#.h5" where "#" is the smallest positive integer that
    # the corresponding filename does not exist.
    if filename_log === nothing
        return nothing
    elseif filename_log[end-2:end] == ".h5"
        return filename_log
    else
        for i in 1:10_000
            if ! isfile("$filename_log.iter$i.h5")
                return "$filename_log.iter$i.h5"
            end
        end
        @warn "Cannot find a non-existent filename $filename_log.iter#.h5. Writing to $filename_log.tmp.h5."
        return filename_log * "tmp.h5"
    end
end

mutable struct mfRGLinearMap{ST, Q} <: LinearMaps.LinearMap{Q}
    const S :: ST
    strategy :: Symbol
    is_first_iteration :: Bool
    function mfRGLinearMap(S :: ST, strategy) where {ST <: AbstractSolver{Q}} where {Q}
        if strategy ∉ [:fdPA, :fdPA_new, :fdPA_1loop]
            error("Invalid strategy $strategy. Must be fdPA or fdPA_new or fdPA_1loop.")
        end
        new{ST, Q}(S, strategy, true)
    end
end

Base.size(A::mfRGLinearMap) = (length(A.S.F), length(A.S.F))

function LinearMaps._unsafe_mul!(y, A::mfRGLinearMap, x::AbstractVector)
    S = A.S

    factor = 1e-2

    unflatten!(S.F, x .* factor)
    # set!(S.F.γa.K1, 0)  # DEBUG mfRG wo K1
    # set!(S.F.γp.K1, 0)  # DEBUG mfRG wo K1
    # set!(S.F.γt.K1, 0)  # DEBUG mfRG wo K1
    # set!(S.F.γa.K2, 0)  # DEBUG mfRG wo K2
    # set!(S.F.γp.K2, 0)  # DEBUG mfRG wo K2
    # set!(S.F.γt.K2, 0)  # DEBUG mfRG wo K2

    build_K3_cache_mfRG!(S, A.is_first_iteration)
    A.is_first_iteration = false

    if A.strategy === :fdPA || A.strategy === :fdPA_1loop
        # Takes ~40% of the time
        BSE_L_K2!(S, pCh)
        BSE_L_K2!(S, aCh)
        BSE_L_K2!(S, tCh)

        BSE_K1!(S, pCh, Val(true))
        BSE_K1!(S, aCh, Val(true))
        BSE_K1!(S, tCh, Val(true))

        # Takes ~60% of the time
        BSE_K2!(S, pCh, Val(true))
        BSE_K2!(S, aCh, Val(true))
        BSE_K2!(S, tCh, Val(true))
    elseif A.strategy === :fdPA_new
        # New version
        BSE_K1_new!(S, pCh, Val(true))  # DEBUG mfRG wo K1
        BSE_K1_new!(S, aCh, Val(true))  # DEBUG mfRG wo K1
        BSE_K1_new!(S, tCh, Val(true))  # DEBUG mfRG wo K1

        BSE_K2_new!(S, pCh, Val(true))  # DEBUG mfRG wo K2
        BSE_K2_new!(S, aCh, Val(true))  # DEBUG mfRG wo K2
        BSE_K2_new!(S, tCh, Val(true))  # DEBUG mfRG wo K2
    end

    BSE_L_K3!(S, pCh)
    BSE_L_K3!(S, aCh)
    BSE_L_K3!(S, tCh)

    BSE_K3!(S, pCh, Val(true))
    BSE_K3!(S, aCh, Val(true))
    BSE_K3!(S, tCh, Val(true))

    set!(S.F, S.Fbuff)

    flatten!(S.F, y)
    y .= x .- y ./ factor

    return y
end


# fixed-point equation
function fixed_point_preconditioned!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver{Q},
    krylov_solver = nothing,
    ;
    strategy,
    update_Σ = false,
    occ_target = nothing,
    hubbard_params = nothing,
    filename_log :: Union{Nothing, String} = nothing,
    use_preconditioner = true,
    krylov_maxiter = 400,
    callback_after_iterate = nothing,
    kwargs_solver...
    ) :: Nothing where {Q}

    # update solver from input vector
    if update_Σ
        unflatten!(S, x)

        if occ_target !== nothing
            # Update chemical potential to fix the occupation
            μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
            set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
        end
    else
        unflatten!(S.F, x)
    end
    symmetrize_solver!(S; verbose = false)

    # Iterate the solver
    iterate_solver!(S; strategy, update_Σ, kwargs_solver...)

    if callback_after_iterate !== nothing
        callback_after_iterate(S)
    end

    if filename_log !== nothing
        if mpi_ismain()
            h5open(unique_filename_log(filename_log), "w") do f
                save!(f, "S", S)
            end
        end
    end

    # calculate residue
    R_F = flatten(S.F) .- x[1:length(S.F)]

    # Precondition output
    if use_preconditioner
        if krylov_solver === nothing
            xsol, stats = Krylov.dqgmres(mfRGLinearMap(S, strategy), R_F;
                atol = 1e-6, rtol = 1e-6, itmax = krylov_maxiter,
                memory = 100, verbose = 0, history = true);
        else
            # In-place solver
            res = Krylov.dqgmres!(krylov_solver, mfRGLinearMap(S, strategy), R_F;
                atol = 1e-6, rtol = 1e-6, itmax = krylov_maxiter,
                verbose = 0, history = true);
            xsol = res.x
            stats = res.stats
        end

        if !stats.solved
            @warn "Krylov solver did not converge after $(stats.niter) iterations, residual $(stats.residuals[end])."
        end
        flush(stdout)
        flush(stderr)

        R[1:length(S.F)] .= xsol
    else
        R[1:length(S.F)] .= R_F
    end

    if update_Σ
        R[length(S.F)+1:end] .= flatten(S.Σ) .- x[length(S.F)+1:end]
    end

    return nothing
end

function iterate_solver_self_energy!(S;
    strategy,
    occ_target = nothing,
    hubbard_params = nothing,
    )

    if occ_target !== nothing
        # Update chemical potential to fix the occupation
        μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
        mpi_ismain() && println("Updated μ = $μ")
        set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
    end

    Dyson!(S)
    bubbles!(S)
    SDE!(S; strategy)
end

function fixed_point_self_energy!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver{Q}
    ;
    filename_log :: Union{Nothing, String} = nothing,
    kwargs_solver...
    ) :: Nothing where {Q}

    # update self-energy from input vector
    unflatten!(S.Σ, x)

    # Iterate the solver
    iterate_solver_self_energy!(S; kwargs_solver...)

    # calculate residue
    flatten!(S.Σ, R)
    R .-= x

    return nothing
end


"""
# Adaptive mixing
Adaptively tune mixing to converge the fdDΓA calculation.
Set ``Π = mixing * Π(G) + mixing * Π(G0)`` and run fdDΓA.
If fdDΓA is converged, increase mixing by 20% and proceed.
If fdDΓA is not converged, reduce mixing by 50% and retry.
"""
function solve_using_mfRG!(
    S :: AbstractSolver
    ;
    filename_log = nothing,
    maxiter = 100,
    verbose = true,
    occ_target = nothing,
    hubbard_params = nothing,
    mixing_init = 1.0,
    iter_restart = 0,
    tol = 1e-4,
    auto_restart = false,
    _debug_single_iter = false,
    strategy = :fdPA,
    update_Σ = false,
    )

    if auto_restart && filename_log !== nothing
        iter = findlast(i -> isfile("$filename_log.iter$i.h5"), 1:100)
        if iter !== nothing
            iter_restart = iter
        end
    end

    if iter_restart !== 0
        # Restart from file
        iter = iter_restart
        filename = "$filename_log.iter$iter_restart.h5"
        mpi_ismain() && println("Restarting from $filename")
        f = h5open(filename, "r")
        set!(S.Gbare,  load_mesh_function(f, "Gbare"))
        set!(S.G,  load_mesh_function(f, "G"))
        set!(S.Σ,  load_mesh_function(f, "Σ"))
        set!(S.G0, load_mesh_function(f, "G0"))
        set!(S.Σ0, load_mesh_function(f, "Σ0"))
        set!(S.Π0pp, load_mesh_function(f, "Π0pp"))
        set!(S.Π0ph, load_mesh_function(f, "Π0ph"))
        set!(S.Πpp, load_mesh_function(f, "Πpp"))
        set!(S.Πph, load_mesh_function(f, "Πph"))
        set!(S.F,  load_vertex(NL2_Vertex, f, "F"))
        if S.F0 isa NL2_Vertex
            set!(S.F0, load_vertex(NL2_Vertex, f, "F0"))
        elseif S.F0 isa NL2_MBEVertex
            set!(S.F0, load_vertex(NL2_MBEVertex, f, "F0"))
        else
            error("Wrong type of S.F0")
        end
        mixing = read(f, "mixing")
        close(f)
    else
        mixing = mixing_init
        iter = 0
    end


    for iter_ in 1:maxiter
        iter += 1

        # Mix bubble
        # Here, S.Π are the unmixed bubble, and S.Π0 is the mixed bubble from the previous iteration.
        Πpp_mixed = S.Πpp * mixing + S.Π0pp * (1 - mixing)
        Πph_mixed = S.Πph * mixing + S.Π0ph * (1 - mixing)
        set!(S.Πpp, Πpp_mixed)
        set!(S.Πph, Πph_mixed)

        if mpi_ismain() && verbose
            println(" === Iteration $iter (including retry: $iter_) ===")
            println("Mix Π and Π0 by $mixing")
            println("max|Πpp - Π0pp| = $(norm((S.Πpp - S.Π0pp).data))")
            println("max|Πph - Π0ph| = $(norm((S.Πph - S.Π0ph).data))")
        end

        # Solve vertex
        if mpi_ismain() && verbose
            println("Solving vertex")
        end
        @time res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S; update_Σ, strategy), flatten(S.F),
            method = :anderson,
            iterations = 40,
            ftol = tol,
            beta = 0.85,
            m = 50,
            show_trace = mpi_ismain() && verbose,
        );
        if _debug_single_iter
            unflatten!(S.F, res.zero);
            return
        end

        if ! res.f_converged
            # If the calculation did not converge, reduce mixing and retry.
            mixing /= 2.0
            iter -= 1
            bubbles!(S)  # Reset bubbles to the unmixed value

            mpi_ismain() && println("Calculation did not converge in $(res.iterations) iterations. Reduce mixing to $mixing")
            continue
        else
            # If the calculation converged, increase mixing.
            mixing = min(1.0, mixing * 1.2)
            if mpi_ismain() && verbose
                println("Calculation converged, increase mixing to $mixing")
            end
        end
        unflatten!(S.F, res.zero);

        # Restore unmixed bubble for SDE
        bubbles_real_space!(S.Π0pp, S.Π0ph, S.G0)
        bubbles_real_space!(S.Πpp, S.Πph, S.G)

        # Update self-energy
        # SDE!(S; strategy = :fdPA)
        SDE!(S; strategy = :scPA)
        Σ_err = absmax(S.Σ - S.Σ0) / mixing


        # Update reference Green function, bubble, and self-energy
        set!(S.Π0pp, Πpp_mixed)
        set!(S.Π0ph, Πph_mixed)
        set!(S.G0, S.G)
        set!(S.Σ0, S.Σ)

        # Update reference vertex and reset target vertex
        add!(S.F0, S.F)
        set!(S.F, 0)

        # Update target Green function and bubble
        if occ_target !== nothing
            # Update chemical potential to fix the occupation
            mpi_ismain() && println("Current occupation $(compute_occupation(S.G)), target $occ_target")
            μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
            set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
        end

        Dyson!(S)
        bubbles!(S)

        flush(stdout)
        flush(stderr)

        if filename_log !== nothing
            if mpi_ismain()
                h5open(filename_log * ".iter$iter.h5", "w") do f
                    save!(f, "S", S)
                    f["mixing"] = mixing
                end
            end
        end


        if mpi_ismain()
            println("Self-energy error: $Σ_err")

            if Σ_err < tol
                break
            end
        end
    end
end





# ---------------------------------------------------------------------------------------
function iterate_solver_self_energy_new!(S;
    strategy,
    occ_target = nothing,
    hubbard_params = nothing,
    Σ_corr,
    )

    if occ_target !== nothing
        # Update chemical potential to fix the occupation
        μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
        mpi_ismain() && println("Updated μ = $μ")
        set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
    end

    Dyson!(S)
    bubbles!(S)

    SDE!(S; strategy)
    add!(S.Σ, Σ_corr)
end

function fixed_point_self_energy_new!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver{Q}
    ;
    kwargs_solver...
    ) :: Nothing where {Q}

    # update self-energy from input vector
    unflatten!(S.Σ, x)

    # Iterate the solver
    iterate_solver_self_energy_new!(S; kwargs_solver...)

    # calculate residue
    flatten!(S.Σ, R)
    R .-= x

    return nothing
end

"""
# Adaptive mixing
Adaptively tune mixing to converge the fdDΓA calculation.
Set ``Π = mixing * Π(G) + mixing * Π(G0)`` and run fdDΓA.
If fdDΓA is converged, increase mixing by 20% and proceed.
If fdDΓA is not converged, reduce mixing by 50% and retry.
"""
function solve_using_mfRG_fixed_bubble!(
    S :: AbstractSolver
    ;
    maxiter = 100,
    verbose = true,
    mixing_step_init = 1.0,
    tol = 1e-3,
    strategy,
    )

    # Converge for given Π and Π0 by incremental mixing
    Πpp = copy(S.Πpp)
    Πph = copy(S.Πph)
    Π0pp = copy(S.Π0pp)
    Π0ph = copy(S.Π0ph)

    mixing_prev = 0.0
    mixing_step = mixing_step_init

    for iter in 1:maxiter

        # Mix bubble
        mixing = min(1.0, mixing_prev + mixing_step)
        set!(S.Πpp, Πpp * mixing + Π0pp * (1 - mixing))
        set!(S.Πph, Πph * mixing + Π0ph * (1 - mixing))

        if mpi_ismain() && verbose
            println(" === Iteration $iter, mixing $mixing_prev -> $mixing ===")
            println("norm|Πpp - Π0pp| = $(norm((S.Πpp - S.Π0pp).data))")
            println("norm|Πph - Π0ph| = $(norm((S.Πph - S.Π0ph).data))")
        end

        # Solve vertex
        mpi_ismain() && verbose && println("Solving vertex")

        kwargs_solver_vertex = (; update_Σ = false, strategy)
        @time res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S; kwargs_solver_vertex...), flatten(S.F),
            method = :anderson,
            iterations = 40,
            ftol = tol,
            beta = 0.85,
            m = 100,
            show_trace = mpi_ismain() && verbose,
        );

        if ! res.f_converged
            # If the calculation did not converge, reduce mixing and retry.
            mixing = mixing_prev
            mixing_step /= 2.0
            set!(S.F, 0)

            mpi_ismain() && println("Calculation did not converge in $(res.iterations) iterations. Reduce mixing_step to $mixing_step")
            continue

        else
            # If the calculation converged, increase mixing step size.
            mixing_prev = mixing
            mixing_step *= 1.2
            mpi_ismain() && println("Calculation converged, increase mixing_step to $mixing_step")
        end

        unflatten!(S.F, res.zero);

        # Update reference vertex and bubble
        add!(S.F0, S.F)
        set!(S.Π0pp, S.Πpp)
        set!(S.Π0ph, S.Πph)

        # Reset target vertex
        set!(S.F, 0)

        if mixing == 1.0
            break
        end
    end

    # Reset bubbles to the unmixed value
    set!(S.Πpp, Πpp)
    set!(S.Πph, Πph)
    set!(S.Π0pp, Π0pp)
    set!(S.Π0ph, Π0ph)

    return nothing
end


function solve_using_mfRG_v2!(
    S :: AbstractSolver
    ;
    filename_log = nothing,
    maxiter = 100,
    verbose = true,
    occ_target = nothing,
    hubbard_params = nothing,
    mixing_init = 1.0,
    iter_restart = 0,
    tol = 1e-4,
    auto_restart = false,
    Σ_corr,
    )

    if auto_restart && filename_log !== nothing
        iter = findlast(i -> isfile("$filename_log.iter$i.h5"), 1:100)
        if iter !== nothing
            iter_restart = iter
        end
    end

    if iter_restart !== 0
        # Restart from file
        iter = iter_restart
        filename = "$filename_log.iter$iter_restart.h5"
        mpi_ismain() && println("Restarting from $filename")
        f = h5open(filename, "r")
        set!(S.Gbare,  load_mesh_function(f, "Gbare"))
        set!(S.G,  load_mesh_function(f, "G"))
        set!(S.Σ,  load_mesh_function(f, "Σ"))
        set!(S.G0, load_mesh_function(f, "G0"))
        set!(S.Σ0, load_mesh_function(f, "Σ0"))
        set!(S.Π0pp, load_mesh_function(f, "Π0pp"))
        set!(S.Π0ph, load_mesh_function(f, "Π0ph"))
        set!(S.Πpp, load_mesh_function(f, "Πpp"))
        set!(S.Πph, load_mesh_function(f, "Πph"))
        set!(S.F,  load_vertex(NL2_Vertex, f, "F"))
        if S.F0 isa NL2_Vertex
            set!(S.F0, load_vertex(NL2_Vertex, f, "F0"))
        elseif S.F0 isa NL2_MBEVertex
            set!(S.F0, load_vertex(NL2_MBEVertex, f, "F0"))
        else
            error("Wrong type of S.F0")
        end
        mixing = read(f, "mixing")
        close(f)
    else
        mixing = mixing_init
        iter = 0
    end


    for iter_ in 1:maxiter
        iter += 1

        # Mix bubble
        # Here, S.Π are the unmixed bubble, and S.Π0 is the mixed bubble from the previous iteration.
        Πpp_mixed = S.Πpp * mixing + S.Π0pp * (1 - mixing)
        Πph_mixed = S.Πph * mixing + S.Π0ph * (1 - mixing)
        set!(S.Πpp, Πpp_mixed)
        set!(S.Πph, Πph_mixed)

        if mpi_ismain() && verbose
            println(" === Iteration $iter (including retry: $iter_) ===")
            println("Mix Π and Π0 by $mixing")
            println("max|Πpp - Π0pp| = $(absmax(S.Πpp - S.Π0pp))")
            println("max|Πph - Π0ph| = $(absmax(S.Πph - S.Π0ph))")
        end

        # Solve vertex
        if mpi_ismain() && verbose
            println("Solving vertex")
        end
        kwargs_solver_vertex = (; update_Σ = false, strategy = :fdPA)
        @time res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S; kwargs_solver_vertex...), flatten(S.F),
            method = :anderson,
            iterations = 200,
            ftol = tol,
            beta = 0.85,
            m = 100,
            show_trace = mpi_ismain() && verbose,
        );

        if ! res.f_converged
            # If the calculation did not converge, reduce mixing and retry.
            mixing /= 2.0
            iter -= 1
            bubbles!(S)  # Reset bubbles to the unmixed value

            mpi_ismain() && println("Calculation did not converge in $(res.iterations) iterations. Reduce mixing to $mixing")
            continue
        else
            # If the calculation converged, increase mixing.
            mixing = min(1.0, mixing * 1.2)
            mpi_ismain() && println("Calculation converged, increase mixing to $mixing")
        end
        unflatten!(S.F, res.zero);

        set!(S.Π0pp, S.Πpp)
        set!(S.Π0ph, S.Πph)

        # ---------------------------------------------------------------------
        # Update self-energy

        verbose = true

        kwargs_solver_selfen = (; strategy = :scPA, occ_target, hubbard_params, Σ_corr)
        @time res = fdDGAsolver.nlsolve((R, x) -> fixed_point_self_energy_new!(R, x, S; kwargs_solver_selfen...), flatten(S.Σ),
            method = :anderson,
            iterations = 100,
            ftol = 1e-5,
            beta = 0.85,
            m = 10,
            show_trace = mpi_ismain() && verbose,
        );
        unflatten!(S.Σ, res.zero);
        Σ_err = absmax(S.Σ - S.Σ0) / mixing

        # Update reference vertex and reset target vertex
        add!(S.F0, S.F)
        set!(S.F, 0)

        # Update self-energy
        set!(S.Σ, S.Σ * 0.5 + S.Σ0 * 0.5)

        set!(S.Σ0, S.Σ)
        set!(S.G0, S.G)

        # Update target Green function and bubble
        if occ_target !== nothing
            # Update chemical potential to fix the occupation
            Dyson!(S)
            mpi_ismain() && println("Current occupation $(compute_occupation(S.G)), target $occ_target")
            μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
            set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
        end

        Dyson!(S)
        bubbles!(S)

        flush(stdout)
        flush(stderr)

        if filename_log !== nothing
            if mpi_ismain()
                h5open(filename_log * ".iter$iter.h5", "w") do f
                    save!(f, "S", S)
                    f["mixing"] = mixing
                end
            end
        end

        if mpi_ismain()
            println("Self-energy error: $Σ_err")

            if Σ_err < tol
                break
            end
        end
    end
end





function solve_using_mfRG_without_mixing!(
    S :: AbstractSolver
    ;
    verbose = true,
    occ_target = nothing,
    hubbard_params = nothing,
    tol = 1e-4,
    strategy = :fdPA,
    update_Σ = false,
    maxiter = 100,
    memory = 50,
    filename_log = nothing,
    use_preconditioner = true,
    store_trace = false,
    mixing = 0.85,
    _debug_outer = 1,
    compute_Hartree = true,
    krylov_maxiter = 400,
    krylov_memory = 200,
    callback_after_iterate = nothing,
    )

    if occ_target !== nothing
        if hubbard_params === nothing
            throw(ArgumentError("hubbard_params must be provided if occ_target is given."))
        end
    end

    if mpi_ismain() && verbose
        println("Solve using mfRG preconditioning, strategy $strategy, tol $tol, maxiter $maxiter, update_Σ $update_Σ")
        println("Parallelization mode: $(S.mode), nthreads = $(Threads.nthreads())")
        if occ_target !== nothing
            println("Occupation fixed to $occ_target, hubbard parameters $hubbard_params")
        end
        println("norm|Πpp - Π0pp| = $(norm((S.Πpp - S.Π0pp).data))")
        println("norm|Πph - Π0ph| = $(norm((S.Πph - S.Π0ph).data))")
    end

    # Solve vertex
    if mpi_ismain() && verbose
        println("Solving vertex")
    end

    res = nothing
    history = NLsolve.SolverState[]

    for i in 1:_debug_outer

        if update_Σ
            x0 = flatten(S)
        else
            x0 = flatten(S.F)
        end

        krylov_solver = Krylov.DqgmresSolver(mfRGLinearMap(S, strategy), flatten(S.F), krylov_memory)

        res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S, krylov_solver; update_Σ, strategy, occ_target, hubbard_params, filename_log, use_preconditioner, compute_Hartree, krylov_maxiter, callback_after_iterate), x0,
            method = :anderson,
            iterations = maxiter,
            ftol = tol,
            beta = mixing,
            m = memory,
            show_trace = mpi_ismain() && verbose,
            store_trace = store_trace,
        );

        if ! res.f_converged
            mpi_ismain() && println("Calculation did not converge in $(res.iterations) iterations.")
        else
            # If the calculation converged, increase mixing.
            if mpi_ismain() && verbose
                println("Calculation converged")
            end
        end

        if update_Σ
            unflatten!(S, res.zero);
        else
            unflatten!(S.F, res.zero);
        end

        symmetrize_solver!(S; verbose = false)
        append!(history, res.trace.states)
    end

    return (; res, history)
end
