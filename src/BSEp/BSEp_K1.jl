# Note: K1 contributions to left and right part always vanish
function BSE_K1!(
    S :: ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω       = wtpl[1]
        val     = zero(Q)
        Π0slice = view(S.Π0pp, Ω, :)
        Πslice  = view(S.Πpp, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0pp, Val(2))[i])

            # vertices
            Fl  = S.F( Ω, νInf, ω, pCh, pSp)
            F0r = S.F0(Ω, Ω - ω, νInf, pCh, pSp)
            FLr = S.FL(Ω, Ω - ω, νInf, pCh, pSp)

            # 1ℓ and central part
            val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
        end

        return temperature(S) * val
    end

    # compute K1
    S.SGpp[1](S.Fbuff.γp.K1, InitFunction{1, Q}(diagram); mode = S.mode)

    return nothing
end


function BSE_K1_mfRG!(
    S :: ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω       = wtpl[1]
        val     = zero(Q)
        Π0slice = view(S.Π0pp, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0pp, Val(2))[i])

            # vertices
            Fl  = S.F0(Ω, νInf, ω, pCh, pSp)
            FLr = S.FL(Ω, Ω - ω, νInf, pCh, pSp)

            # 1ℓ and central part
            val += Fl * Π0slice[i] * FLr
        end

        return temperature(S) * val
    end

    # compute K1
    S.SGpp[1](S.Fbuff.γp.K1, InitFunction{1, Q}(diagram); mode = S.mode)

    return nothing
end
