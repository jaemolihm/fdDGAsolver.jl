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
            ω = value(meshes(S.Π0ph, Val(2))[i])

            # vertices
            Γp  = S.F(Ω, ν, ω, aCh, pSp; F0 = false, γa = false)
            F0p = S.F0(Ω, ω, νInf, aCh, pSp)

            val += Γp * Π0slice[i] * F0p
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
            ω = value(meshes(S.Π0ph, Val(2))[i])

            # vertices
            Fl  = S.F(Ω, ν, ω, aCh, pSp)
            F0r = S.F0(Ω, ω, νInf, aCh, pSp)
            FLr = S.FL(Ω, ω, νInf, aCh, pSp)

            # 1ℓ and central part
            val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGph[2](S.Fbuff.γa.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γa.K2, S.FL.γa.K2)

    return nothing
end
