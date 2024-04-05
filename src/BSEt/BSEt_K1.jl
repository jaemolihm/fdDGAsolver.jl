# Note: K1 contributions to left and right part always vanish
function BSE_K1!(
    S :: ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω       = wtpl[1]
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :)
        Πslice  = view(S.Πph, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, 2)[i])

            # vertices
            Fdl  = S.F(Ω, νInf, ω, tCh, dSp; γp = false, γa = false)
            F0dr = S.F0(Ω, ω, νInf, tCh, dSp; γp = false, γa = false)

            # 1ℓ part
            val -= (Πslice[i] - Π0slice[i]) * Fdl * F0dr

            # central part
            val -= Πslice[i] * Fdl * (2 * box_eval(S.FL.γt.K2, Ω, ω) - box_eval(S.FL.γa.K2, Ω, ω))
        end

        return temperature(S) * val
    end

    # compute K1
    S.SGph[1](S.Fbuff.γt.K1, InitFunction{1, Q}(diagram); mode = S.mode)

    # Currently S.Fbuff.γt.K1 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K1, S.Fbuff.γa.K1)
    S.Fbuff.γt.K1.data ./= 2

    return nothing
end
