# Note: K1 contributions to left and right part always vanish
function BSE_K1!(
    S :: NL_ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P)
        Πslice  = view(S.Πph , Ω, :, P)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, Val(2))[i])

            # vertices
            Fl  = S.F( Ω, νInf, ω, P, kSW, kSW, aCh, pSp)
            F0r = S.F0(Ω, ω, νInf, P, kSW, kSW, aCh, pSp)
            FLr = S.FL(Ω, ω, νInf, P, kSW, kSW, aCh, pSp)

            # 1ℓ and central part
            val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
        end

        return temperature(S) * val
    end

    # compute K1
    S.SGph[1](S.Fbuff.γa.K1, InitFunction{2, Q}(diagram); mode = S.mode)

    return nothing
end


# Note: K1 contributions to left and right part always vanish
function BSE_K1_mfRG!(
    S :: NL_ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, Val(2))[i])

            # vertices
            Fl  = S.F0(Ω, νInf, ω, P, kSW, kSW, aCh, pSp)
            FLr = S.FL(Ω, ω, νInf, P, kSW, kSW, aCh, pSp)

            # 1ℓ and central part
            val += Fl * Π0slice[i] * FLr
        end

        return temperature(S) * val
    end

    # compute K1
    S.SGph[1](S.Fbuff.γa.K1, InitFunction{2, Q}(diagram); mode = S.mode)

    return nothing
end
