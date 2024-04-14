function BSE_L_K2!(
    S :: NL_ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, Val(2))[i.I[1]])
            q = value(meshes(S.Π0ph, Val(4))[i.I[2]])

            # vertices
            Γp  = S.F( Ω, ν,    ω, P, kSW,  q, aCh, pSp; F0 = false, γa = false)
            F0p = S.F0(Ω, ω, νInf, P,   q, k0, aCh, pSp)

            val += Γp * Π0slice[i] * F0p
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGphL[2](S.FL.γa.K2, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    S :: NL_ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P, :)
        Πslice  = view(S.Πph , Ω, :, P, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, Val(2))[i.I[1]])
            q = value(meshes(S.Π0ph, Val(4))[i.I[2]])

            # vertices
            Fl  = S.F( Ω, ν,    ω, P, kSW,  q, aCh, pSp) - S.F(Ω, νInf, ω, P, kSW, q, aCh, pSp)
            F0r = S.F0(Ω, ω, νInf, P,   q, k0, aCh, pSp)
            FLr = S.FL(Ω, ω, νInf, P,   q, k0, aCh, pSp)

            # 1ℓ and central part
            val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGph[2](S.Fbuff.γa.K2, InitFunction{3, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γa.K2, S.FL.γa.K2)

    return nothing
end
