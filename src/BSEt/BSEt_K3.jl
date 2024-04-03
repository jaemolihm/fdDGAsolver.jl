function BSE_L_K3!(
    S  :: ParquetSolver{Q},
    Γt  :: MF_K3{Q},
    Γa  :: MF_K3{Q},
    F0t :: MF_K3{Q},
    F0a :: MF_K3{Q},
       :: Type{tCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Π0slice  = view(S.Π0ph, Ω, :)
        Γtslice  = view(Γt, Ω, ν, :)
        Γaslice  = view(Γa, Ω, ν, :)
        F0tslice = view(F0t, Ω, :, νp)
        F0aslice = view(F0a, Ω, :, νp)

        # additional minus sign for xSp terms because we use crossing symmetry here
        for i in 1 : length(meshes(Γt, 3))
            val -= Π0slice[i] * ((2.0 * Γtslice[i] - Γaslice[i]) * F0tslice[i] - Γtslice[i] * F0aslice[i])
        end

        return temperature(S) * val
    end

    # compute K3
    S.SGphL[3](S.FL.γt.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end

function BSE_K3!(
    S  :: ParquetSolver{Q},
    Γt :: MF_K3{Q},
    Γa :: MF_K3{Q},
    Ft :: MF_K3{Q},
    Fa :: MF_K3{Q},
    F0t :: MF_K3{Q},
    F0a :: MF_K3{Q},
       :: Type{tCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Π0slice  = view(S.Π0ph, Ω, :)
        Πslice   = view(S.Πph, Ω, :)
        Γtslice  = view(Γt, Ω, νp, :)
        Γaslice  = view(Γa, Ω, νp, :)
        Ftslice  = view(Ft, Ω, ν, :)
        Faslice  = view(Fa, Ω, ν, :)
        F0tslice = view(F0t, Ω, :, νp)
        F0aslice = view(F0a, Ω, :, νp)

        # vectorize 1ℓ and right part, additional minus sign for xSp terms because we use crossing symmetry here
        for i in 1 : length(meshes(Γt, 3))
            val -= (Πslice[i] - Π0slice[i]) * ((2 * Ftslice[i] - Faslice[i]) * F0tslice[i] - Ftslice[i] * F0aslice[i]) +
                    Πslice[i] * ((2 * Ftslice[i] - Faslice[i]) * Γtslice[i] - Ftslice[i] * Γaslice[i])
        end

        for i in eachindex(meshes(Γt, 3))
            ω = value(meshes(Γt, 3)[i])

            # vertices, additional minus sign for xSp terms because we use crossing symmetry here
            Fpl = Ftslice[i]
            Fxl = -Faslice[i]

            # central part
            if is_inbounds(ω, meshes(S.FL.γt.K3, 2)) && is_inbounds(νp, meshes(S.FL.γt.K3, 3))
                val -= Πslice[i] * ((2 * Fpl + Fxl) * S.FL.γt.K3[Ω, ω, νp] - Fpl * S.FL.γa.K3[Ω, ω, νp])
            elseif is_inbounds(ω, meshes(S.FL.γt.K2, 2))
                val -= Πslice[i] * ((2 * Fpl + Fxl) * S.FL.γt.K2[Ω, ω] - Fpl * S.FL.γa.K2[Ω, ω])
            # else
            #     val -= Πslice[i] * ((2 * Fpl + Fxl) * box_eval(S.FL.γt.K2, Ω, ω) - Fpl * box_eval(S.FL.γa.K2, Ω, ω))
            end
        end

        return S.FL.γt.K3[Ω, ν, νp] + temperature(S) * val
    end

    # compute K3
    S.SGph[3](S.Fbuff.γt.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end