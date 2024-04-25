mutable struct mfRGLinearMap{ST, Q} <: LinearMaps.LinearMap{Q}
    const S :: ST
    is_first_iteration :: Bool
    function mfRGLinearMap(S :: ST) where {ST <: AbstractSolver{Q}} where {Q}
        new{ST, Q}(S, true)
    end
end

Base.size(A::mfRGLinearMap) = (length(A.S.F), length(A.S.F))

function LinearMaps._unsafe_mul!(y, A::mfRGLinearMap, x::AbstractVector)
    S = A.S

    unflatten!(S.F, x)

    build_K3_cache_mfRG!(S, A.is_first_iteration)
    A.is_first_iteration = false

    # Takes ~40% of the time
    BSE_L_K2!(S, pCh)
    BSE_L_K2!(S, aCh)
    BSE_L_K2!(S, tCh)

    BSE_L_K3!(S, pCh)
    BSE_L_K3!(S, aCh)
    BSE_L_K3!(S, tCh)

    set!(S.Fbuff, 0)

    BSE_K1_mfRG!(S, pCh)
    BSE_K1_mfRG!(S, aCh)
    BSE_K1_mfRG!(S, tCh)

    # Takes ~60% of the time
    BSE_K2_mfRG!(S, pCh)
    BSE_K2_mfRG!(S, aCh)
    BSE_K2_mfRG!(S, tCh)

    BSE_K3_mfRG!(S, pCh)
    BSE_K3_mfRG!(S, aCh)
    BSE_K3_mfRG!(S, tCh)

    set!(S.F, S.Fbuff)

    flatten!(S.F, y)
    y .-= x

    return y
end


# fixed-point equation
function fixed_point_preconditioned!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver{Q}
    ;
    kwargs_solver...
    ) :: Nothing where {Q}

    # update solver from input vector
    unflatten!(S.F, x)

    # Iterate the solver
    iterate_solver!(S; kwargs_solver...)

    # calculate residue
    flatten!(S.F, R)
    R .-= x

    # Precondition output
    res = Krylov.dqgmres(mfRGLinearMap(S), .-R;
        atol = 1e-6, rtol = 1e-6, itmax = 400,
        memory = 200, verbose = 0, history = true);

    if !res[2].solved
        @warn "Krylov solver did not converge after $(res[2].niter) iterations, residual $(res[2].residuals[end])."
    end
    flush(stdout)
    flush(stderr)

    R .= res[1]

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
        set!(S.F0, load_vertex(NL2_Vertex, f, "F0"))
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
            iterations = 100,
            ftol = tol,
            beta = 0.85,
            m = 50,
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

        # Restore unmixed bubble for SDE
        bubbles_real_space!(S.Π0pp, S.Π0ph, S.G0)
        bubbles_real_space!(S.Πpp, S.Πph, S.G)

        # Update self-energy
        SDE!(S; strategy = :fdPA)
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
