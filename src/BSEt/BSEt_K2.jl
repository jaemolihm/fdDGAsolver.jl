function BSE_L_K2!(
    S :: ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, 2)[i])

            # vertices
            Γd  = S.F(Ω, ν, ω, tCh, dSp; F0 = false, γt = false)
            F0d = S.F0(Ω, ω, νInf, tCh, dSp)

            val -= Π0slice[i] * Γd * F0d
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGphL[2](S.FL.γt.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    # Currently S.FL.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.FL.γt.K2, S.FL.γa.K2)
    S.FL.γt.K2.data ./= 2

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    S :: ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :)
        Πslice  = view(S.Πph, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, 2)[i])

            # vertices
            Fl  = S.F(Ω, ν, ω, tCh, dSp)
            F0r = S.F0(Ω, ω, νInf, tCh, dSp)
            FLr = S.FL(Ω, ω, νInf, tCh, dSp)

            # 1ℓ and central part
            val -= Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGph[2](S.Fbuff.γt.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γt.K2, S.FL.γt.K2, 2)
    add!(S.Fbuff.γt.K2, S.FL.γa.K2, -1)

    # Currently S.Fbuff.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K2, S.Fbuff.γa.K2)
    S.Fbuff.γt.K2.data ./= 2

    return nothing
end
