function BSE_L_K2!(
    S :: ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0pp, Ω, :)

        for i in eachindex(Π0slice)
            ω    = value(meshes(S.Π0pp, 2)[i])
            val += S.F(Ω, ν, ω, pCh, pSp; F0 = false, γp = false) * Π0slice[i] * S.F0(Ω, Ω - ω, S.νInf, pCh, pSp; γt = false, γa = false)
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGppL[2](S.FL.γp.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    S :: ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0pp, Ω, :)
        Πslice  = view(S.Πpp, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0pp, 2)[i])

            # vertices
            Fpl  = S.F(Ω, ν, ω, pCh, pSp)
            F0pr = S.F0(Ω, Ω - ω, S.νInf, pCh, pSp; γt = false, γa = false)

            # 1ℓ and central part
            val += Fpl * ((Πslice[i] - Π0slice[i]) * F0pr +  Πslice[i] * box_eval(S.FL.γp.K2, Ω, Ω - ω))
        end

        return S.FL.γp.K2[Ω, ν] + temperature(S) * val
    end

    # compute K2
    S.SGpp[2](S.Fbuff.γp.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    return nothing
end