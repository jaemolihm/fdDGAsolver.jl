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
            Fpl  = S.F(Ω, νInf, ω, tCh, pSp; γp = false, γa = false)
            Fxl  = S.F(Ω, νInf, ω, tCh, xSp; γp = false, γa = false)
            F0pr = S.F0(Ω, ω, νInf, tCh, pSp; γp = false, γa = false)
            F0xr = S.F0(Ω, ω, νInf, tCh, xSp; γp = false, γa = false)

            # 1ℓ part
            val -= (Πslice[i] - Π0slice[i]) * ((2 * Fpl + Fxl) * F0pr + Fpl * F0xr)

            # central part
            val -= Πslice[i] * ((2 * Fpl + Fxl) * box_eval(S.FL.γt.K2, Ω, ω) - Fpl * box_eval(S.FL.γa.K2, Ω, ω))
        end

        return temperature(S) * val
    end

    # compute K1
    S.SGph[1](S.Fbuff.γt.K1, InitFunction{1, Q}(diagram); mode = S.mode)

    return nothing
end
