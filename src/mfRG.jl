struct mfRGLinearMap{ST, Q} <: LinearMaps.LinearMap{Q}
    S :: ST
    function mfRGLinearMap(S :: ST) where {ST <: AbstractSolver{Q}} where {Q}
        new{ST, Q}(S)
    end
end

Base.size(A::mfRGLinearMap) = (length(A.S.F), length(A.S.F))

function LinearMaps._unsafe_mul!(y, A::mfRGLinearMap, x::AbstractVector)
    S = A.S

    unflatten!(S.F, x)

    build_K3_cache_mfRG!(S)

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

    BSE_K2_mfRG!(S, pCh)
    BSE_K2_mfRG!(S, aCh)
    BSE_K2_mfRG!(S, tCh)

    BSE_K3_mfRG!(S, pCh)
    BSE_K3_mfRG!(S, aCh)
    BSE_K3_mfRG!(S, tCh)

    set!(S.F, S.Fbuff)

    y .= flatten(S.F) .- x

    return y
end


# julia> @time mfRGLinearMap(S) * flatten(S.F);
#   0.174776 seconds (123 allocations: 12.672 KiB)
#
#   0.073509 seconds (9.89 k allocations: 624.344 KiB)
#   0.074145 seconds (9.89 k allocations: 624.344 KiB)
#   0.136028 seconds (9.89 k allocations: 624.344 KiB)
#
#   0.006366 seconds (4.58 k allocations: 291.469 KiB)
#   0.006095 seconds (4.58 k allocations: 291.469 KiB)
#   0.005945 seconds (4.58 k allocations: 291.469 KiB)
#
#   0.012447 seconds (702 allocations: 33.219 KiB)
#   0.011846 seconds (702 allocations: 33.219 KiB)
#   0.025590 seconds (702 allocations: 33.219 KiB)
#
#   0.241593 seconds (9.89 k allocations: 624.344 KiB)
#   0.225840 seconds (9.89 k allocations: 624.344 KiB)
#   0.481961 seconds (9.89 k allocations: 624.344 KiB)
#
#   0.003829 seconds (2.72 k allocations: 175.219 KiB)
#   0.003710 seconds (2.72 k allocations: 175.219 KiB)
#   0.003541 seconds (2.72 k allocations: 175.219 KiB)
#   1.539046 seconds (83.96 k allocations: 31.851 MiB, 0.21% gc time)


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
    res = Krylov.dqgmres(mfRGLinearMap(S), R;
        atol = 1e-5, rtol = 1e-5, itmax = 400,
        memory = 400, verbose = 0, history = true);

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
        mpi_ismain() && println("Current occupation $(compute_occupation(S.G)), target $occ_target")
        μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
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

function solve_using_mfRG_mix_bubble!(
    S :: AbstractSolver
    ;
    filename_log = nothing,
    maxiter = 100,
    verbose = true,
    occ_target = nothing,
    hubbard_params = nothing,
    bubble_mixing = 1.0,
    )

    for iter in 1:maxiter
        # Mix bubble
        @. S.Πph.data = S.Πph.data * bubble_mixing + S.Π0ph.data * (1 - bubble_mixing);
        @. S.Πpp.data = S.Πpp.data * bubble_mixing + S.Π0pp.data * (1 - bubble_mixing);

        if mpi_ismain()
            println(" === Iteration $iter ===")
            println("Bubble mixing by $bubble_mixing")
            println("max|Πpp - Π0pp| = $(absmax(S.Πpp - S.Π0pp))")
            println("max|Πph - Π0ph| = $(absmax(S.Πph - S.Π0ph))")
        end

        # Solve vertex
        if mpi_ismain()
            println("Solving vertex")
        end
        kwargs_solver_vertex = (; update_Σ = false, strategy = :fdPA)
        @time res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S; kwargs_solver_vertex...), flatten(S.F),
            method = :anderson,
            iterations = 200,
            ftol = 1e-4,
            beta = 0.85,
            m = 100,
            show_trace = mpi_ismain() && verbose,
        );

        unflatten!(S.F, res.zero);

        # Store current target for the update after SDE
        Πpp_copy = copy(S.Πpp)
        Πph_copy = copy(S.Πph)
        G_copy = copy(S.G)

        SDE!(S; strategy = :fdPA)
        Σ_copy = copy(S.Σ)


        # Solve self-energy
        if mpi_ismain()
            println("Solving self-energy")
        end
        kwargs_solver_selfen = (; strategy = :fdPA, occ_target, hubbard_params)
        @time res = nlsolve((R, x) -> fixed_point_self_energy!(R, x, S; kwargs_solver_selfen...), flatten(S.Σ),
            method = :anderson,
            iterations = 100,
            ftol = 1e-4,
            beta = 0.85,
            m = 10,
            show_trace = mpi_ismain() && verbose,
        );

        unflatten!(S.Σ, res.zero);

        # Update fdPA reference

        # Update reference Green fucntion, bubble, and self-energy
        set!(S.Π0pp, Πpp_copy)
        set!(S.Π0ph, Πph_copy)
        set!(S.G0, G_copy)
        set!(S.Σ0, Σ_copy)

        # Update reference vertex and reset target vertex
        add!(S.F0, S.F)
        set!(S.F, 0)

        # Update target Green function and bubble
        Dyson!(S)
        bubbles!(S)

        flush(stdout)
        flush(stderr)

        if filename_log !== nothing
            if mpi_ismain()
                h5open(filename_log * ".iter$iter.h5", "w") do f
                    save!(f, "S", S)
                end
            end
        end


        if mpi_ismain()
            err = absmax(S.Σ - S.Σ0)
            println("Self-energy error: $err")

            if err < 1e-4
                break
            end
        end
    end
end


function solve_using_mfRG_mix_G!(
    S :: AbstractSolver
    ;
    filename_log = nothing,
    maxiter = 100,
    verbose = true,
    occ_target = nothing,
    hubbard_params = nothing,
    mixing_G_init = 1.0,
    )

    mixing_G = mixing_G_init

    iter = 0

    for iter_ in 1:maxiter
        iter += 1

        # Mix Green function
        G_copy = copy(S.G)
        set!(S.G, S.G * mixing_G + S.G0 * (1 - mixing_G))
        bubbles!(S)

        if mpi_ismain()
            println(" === Iteration $iter (including retry: $iter_) ===")
            println("G mixing by $mixing_G")
            println("max|Πpp - Π0pp| = $(absmax(S.Πpp - S.Π0pp))")
            println("max|Πph - Π0ph| = $(absmax(S.Πph - S.Π0ph))")
        end

        # Solve vertex
        if mpi_ismain()
            println("Solving vertex")
        end
        kwargs_solver_vertex = (; update_Σ = false, strategy = :fdPA)
        @time res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S; kwargs_solver_vertex...), flatten(S.F),
            method = :anderson,
            iterations = 150,
            ftol = 1e-4,
            beta = 0.85,
            m = 100,
            show_trace = mpi_ismain() && verbose,
        );

        if ! res.f_converged
            # If the calculation did not converge, reduce mixing and retry.
            mixing_G /= 2.0
            set!(S.F, 0)
            set!(S.G, G_copy)
            iter -= 1
            mpi_ismain() && println("Calculation did not converge in $(res.iterations) iterations. Reduce mixing_G to $mixing_G")
            continue
        else
            # If the calculation converged, increate mixing.
            mixing_G = min(1.0, mixing_G * 1.1)
            mpi_ismain() && println("Calculation converged, increase mixing_G to $mixing_G")
        end
        unflatten!(S.F, res.zero);

        # Update self-energy
        SDE!(S; strategy = :fdPA)
        Σ_err = absmax(S.Σ - S.Σ0) / mixing_G


        # Update reference Green fucntion, bubble, and self-energy
        set!(S.Π0pp, S.Πpp)
        set!(S.Π0ph, S.Πph)
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
                end
            end
        end


        if mpi_ismain()
            println("Self-energy error: $Σ_err")

            if Σ_err < 1e-4
                break
            end
        end
    end
end
