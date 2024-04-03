# Note: K1 contributions to left and right part always vanish
function BSE_K1!(
    S :: ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω       = wtpl[1]
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :)
        Πslice  = view(S.Πph, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, 2)[i])

            # vertices
            Fpl  = S.F(Ω, S.νInf, ω, aCh, pSp; γp = false, γt = false)
            F0pr = S.F0(Ω, ω, S.νInf, aCh, pSp; γp = false, γt = false)

            # 1ℓ part
            val += Fpl * (Πslice[i] - Π0slice[i]) * F0pr

            # central part
            if is_inbounds(Ω, meshes(S.FL.γa.K2, 1)) && is_inbounds(ω, meshes(S.FL.γa.K2, 2))
                val += Fpl * Πslice[i] * S.FL.γa.K2[Ω, ω]
            end
        end

        return temperature(S) * val
    end

    # compute K1
    S.SGph[1](S.Fbuff.γa.K1, InitFunction{1, Q}(diagram); mode = S.mode)

    return nothing
end