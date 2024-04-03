function BSE_L_K2!(
    S :: ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :)

        for i in eachindex(Π0slice)
            ω    = value(meshes(S.Π0ph, 2)[i])
            val += S.F(Ω, ν, ω, aCh, pSp; F0 = false, γa = false) * Π0slice[i] * S.F0(Ω, ω, S.νInf, aCh, pSp; γp = false, γt = false)
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGphL[2](S.FL.γa.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    S :: ParquetSolver{Q},
      :: Type{aCh}
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
            Fpl  = S.F(Ω, ν, ω, aCh, pSp)
            F0pr = S.F0(Ω, ω, S.νInf, aCh, pSp; γp = false, γt = false)

            # 1ℓ part
            val += Fpl * (Πslice[i] - Π0slice[i]) * F0pr

            # central part
            if is_inbounds(ω, meshes(S.FL.γa.K2, 2))
                val += Fpl * Πslice[i] * S.FL.γa.K2[Ω, ω]
            end
        end

        return S.FL.γa.K2[Ω, ν] + temperature(S) * val
    end

    # compute K2
    S.SGph[2](S.Fbuff.γa.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    return nothing
end