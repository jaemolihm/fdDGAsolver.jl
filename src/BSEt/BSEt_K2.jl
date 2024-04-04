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
            Γp  = S.F(Ω, ν, ω, tCh, pSp; F0 = false, γt = false)
            Γx  = S.F(Ω, ν, ω, tCh, xSp; F0 = false, γt = false)
            F0p = S.F0(Ω, ω, νInf, tCh, pSp; γp = false, γa = false)
            F0x = S.F0(Ω, ω, νInf, tCh, xSp; γp = false, γa = false)

            val -= Π0slice[i] * ((2.0 * Γp + Γx) * F0p + Γp * F0x)
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGphL[2](S.FL.γt.K2, InitFunction{2, Q}(diagram); mode = S.mode)

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
            Fpl  = S.F(Ω, ν, ω, tCh, pSp)
            Fxl  = S.F(Ω, ν, ω, tCh, xSp)
            F0pr = S.F0(Ω, ω, νInf, tCh, pSp; γp = false, γa = false)
            F0xr = S.F0(Ω, ω, νInf, tCh, xSp; γp = false, γa = false)

            # 1ℓ part
            val -= (Πslice[i] - Π0slice[i]) * ((2 * Fpl + Fxl) * F0pr + Fpl * F0xr)

            # central part
            if is_inbounds(ω, meshes(S.FL.γt.K2, 2))
                val -= Πslice[i] * ((2 * Fpl + Fxl) * S.FL.γt.K2[Ω, ω] - Fpl * S.FL.γa.K2[Ω, ω])
            end
        end

        return S.FL.γt.K2[Ω, ν] + temperature(S) * val
    end

    # compute K2
    S.SGph[2](S.Fbuff.γt.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    return nothing
end