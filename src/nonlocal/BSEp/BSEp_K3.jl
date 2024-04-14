function BSE_L_K3!(
    S  :: NL_ParquetSolver{Q},
       :: Type{pCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(S.cache_Γpp, Ω, ν,  :, P)
        F0slice = view(S.cache_F0p, Ω, :, νp, P)

        # additional minus sign because we use crossing symmetry here
        for i in eachindex(meshes(S.cache_Γpp, Val(3)))
            ω = value(meshes(S.cache_Γpp, Val(3))[i])
            Π0 = S.Π0pp[Ω, ω, P, kSW]

            val -= Γslice[i] * Π0 * F0slice[i]
        end

        return temperature(S) * val
    end

    # compute K3
    S.SGppL[3](S.FL.γp.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end

function BSE_K3!(
    S :: NL_ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(S.cache_Γpx, Ω, :, νp, P)
        Fslice  = view(S.cache_Fp,  Ω, ν,  :, P)
        F0slice = view(S.cache_F0p, Ω, :, νp, P)

        for i in eachindex(meshes(S.cache_Fp, Val(3)))
            ω  = value(meshes(S.cache_Fp, Val(3))[i])
            Π0 = S.Π0pp[Ω, ω, P, kSW]
            Π  = S.Πpp[ Ω, ω, P, kSW]

            # 1ℓ and right part, additional minus sign because we use crossing symmetry here
            val -= Fslice[i] * ((Π - Π0) * F0slice[i] + Π * Γslice[i])

            # central part
            if is_inbounds(Ω - ω, meshes(S.FL.γp.K3, Val(2)))
                val += Fslice[i] * Π * S.FL.γp.K3[Ω, Ω - ω, νp, P]
            elseif is_inbounds(Ω - ω, meshes(S.FL.γp.K2, Val(2)))
                val += Fslice[i] * Π * S.FL.γp.K2[Ω, Ω - ω, P]
            end
        end

        return S.FL.γp.K3[Ω, ν, νp, P] + temperature(S) * val
    end

    # compute K3
    S.SGpp[3](S.Fbuff.γp.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end
