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
            ω = value(meshes(S.Π0pp, 2)[i])

            # vertices
            Fpl  = S.F(Ω, S.νInf, ω, pCh, pSp; γt = false, γa = false)
            F0pr = S.F0(Ω, Ω - ω, S.νInf, pCh, pSp; γt = false, γa = false)

            # 1ℓ part
            val += Fpl * (Πslice[i] - Π0slice[i]) * F0pr

            # central part
            if is_inbounds(Ω, meshes(S.FL.γp.K2, 1)) && is_inbounds(Ω - ω, meshes(S.FL.γp.K2, 2))
                val += Fpl * Πslice[i] * S.FL.γp.K2[Ω, Ω - ω]
            end
        end

        return temperature(S) * val
    end

    # compute K1
    S.SGpp[1](S.Fbuff.γp.K1, InitFunction{1, Q}(diagram); mode = S.mode)

    return nothing
end