function BSE_L_K3!(
    S  :: ParquetSolver{Q},
    Γt  :: MF_K3{Q},
    F0t :: MF_K3{Q},
       :: Type{tCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Π0slice  = view(S.Π0ph, Ω, :)
        Γtslice  = view(Γt, Ω, ν, :)
        F0tslice = view(F0t, Ω, :, νp)

        # additional minus sign for xSp terms because we use crossing symmetry here
        for i in 1 : length(meshes(Γt, 3))
            val -= Π0slice[i] * Γtslice[i] * F0tslice[i]
        end

        return temperature(S) * val
    end

    # compute K3
    S.SGphL[3](S.FL.γt.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    # Currently S.FL.γt.K3 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.FL.γt.K3, S.FL.γa.K3)
    S.FL.γt.K3.data ./= 2

    return nothing
end

function BSE_K3!(
    S  :: ParquetSolver{Q},
    Γt :: MF_K3{Q},
    Ft :: MF_K3{Q},
    F0t :: MF_K3{Q},
       :: Type{tCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Π0slice  = view(S.Π0ph, Ω, :)
        Πslice   = view(S.Πph, Ω, :)
        Γtslice  = view(Γt, Ω, νp, :)
        Ftslice  = view(Ft, Ω, ν, :)
        F0tslice = view(F0t, Ω, :, νp)

        # vectorize 1ℓ and right part, additional minus sign for xSp terms because we use crossing symmetry here
        for i in 1 : length(meshes(Γt, 3))
            val -= (Πslice[i] - Π0slice[i]) * Ftslice[i] * F0tslice[i] + Πslice[i] * Ftslice[i] * Γtslice[i]
        end

        for i in eachindex(meshes(Γt, 3))
            ω = value(meshes(Γt, 3)[i])

            # vertices, additional minus sign for xSp terms because we use crossing symmetry here
            Fpl = Ftslice[i]

            # central part
            if is_inbounds(ω, meshes(S.FL.γt.K3, 2)) && is_inbounds(νp, meshes(S.FL.γt.K3, 3))
                val -= Πslice[i] * Fpl * (2 * S.FL.γt.K3[Ω, ω, νp] - S.FL.γa.K3[Ω, ω, νp])
            else
                val -= Πslice[i] * Fpl * (2 * S.FL.γt.K2(Ω, ω) - S.FL.γa.K2(Ω, ω))
            end
        end

        return 2 * S.FL.γt.K3[Ω, ν, νp] - S.FL.γa.K3[Ω, ν, νp] + temperature(S) * val
    end

    # compute K3
    S.SGph[3](S.Fbuff.γt.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    # Currently S.Fbuff.γt.K3 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K3, S.Fbuff.γa.K3)
    S.Fbuff.γt.K3.data ./= 2

    return nothing
end
