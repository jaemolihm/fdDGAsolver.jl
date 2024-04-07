function BSE_L_K2!(
    S :: NL2_ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0pp, Ω, :, P, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0pp, 2)[i.I[1]])
            q = value(meshes(S.Π0pp, 4)[i.I[2]])

            # Vertices
            Γp  = S.F( Ω,     ν,    ω, P,     k,  q, pCh, pSp; F0 = false, γp = false)
            F0p = S.F0(Ω, Ω - ω, νInf, P, P - q, k0, pCh, pSp)

            val += Γp * Π0slice[i] * F0p
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGppL[2](S.FL.γp.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    S :: NL2_ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0pp, Ω, :, P, :)
        Πslice  = view(S.Πpp , Ω, :, P, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0pp, 2)[i.I[1]])
            q = value(meshes(S.Π0pp, 4)[i.I[2]])

            # vertices
            Fl  = S.F( Ω,     ν,    ω, P,     k,  q, pCh, pSp)
            F0r = S.F0(Ω, Ω - ω, νInf, P, P - q, k0, pCh, pSp)
            FLr = S.FL(Ω, Ω - ω, νInf, P, P - q, k0, pCh, pSp)

            # 1ℓ and central part
            val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGpp[2](S.Fbuff.γp.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γp.K2, S.FL.γp.K2)

    return nothing
end
