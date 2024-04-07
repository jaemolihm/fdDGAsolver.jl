# Note: K1 contributions to left and right part always vanish
function BSE_K1!(
    S :: Union{NL_ParquetSolver{Q}, NL2_ParquetSolver{Q}},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0pp, Ω, :, P, :)
        Πslice  = view(S.Πpp , Ω, :, P, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0pp, 2)[i.I[1]])
            q = value(meshes(S.Π0pp, 4)[i.I[2]])

            # vertices
            Fl  = S.F( Ω,  νInf,    ω, P,    k0,  q, pCh, pSp)
            F0r = S.F0(Ω, Ω - ω, νInf, P, P - q, k0, pCh, pSp)
            FLr = S.FL(Ω, Ω - ω, νInf, P, P - q, k0, pCh, pSp)

            # 1ℓ and central part
            val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K1
    S.SGpp[1](S.Fbuff.γp.K1, InitFunction{2, Q}(diagram); mode = S.mode)

    return nothing
end
