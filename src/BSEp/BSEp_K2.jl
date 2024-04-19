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
            ω = value(meshes(S.Π0pp, Val(2))[i])

            # Vertices
            Γp  = S.F(Ω, ν, ω, pCh, pSp; F0 = false, γp = false)
            F0p = S.F0(Ω, Ω - ω, νInf, pCh, pSp)

            val += Γp * Π0slice[i] * F0p
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
            ω = value(meshes(S.Π0pp, Val(2))[i])

            # vertices
            Fl  = S.F(Ω, ν, ω, pCh, pSp) - S.F(Ω, νInf, ω, pCh, pSp)
            F0r = S.F0(Ω, Ω - ω, νInf, pCh, pSp)
            FLr = S.FL(Ω, Ω - ω, νInf, pCh, pSp)

            # 1ℓ and central part
            val += Fl * ((Πslice[i] - Π0slice[i]) * F0r +  Πslice[i] * FLr)
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGpp[2](S.Fbuff.γp.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γp.K2, S.FL.γp.K2)

    return nothing
end



# Note: K2 contributions to right part always vanishes
function BSE_K2_mfRG!(
    S :: ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0pp, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0pp, Val(2))[i])

            # vertices
            Fl  = S.F0(Ω, ν, ω, pCh, pSp) - S.F0(Ω, νInf, ω, pCh, pSp)
            FLr = S.FL(Ω, Ω - ω, νInf, pCh, pSp)

            # 1ℓ and central part
            val += Fl * Π0slice[i] * FLr
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGpp[2](S.Fbuff.γp.K2, InitFunction{2, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γp.K2, S.FL.γp.K2)

    return nothing
end
