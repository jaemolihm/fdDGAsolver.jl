function BSE_L_K3!(
    S :: ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Γslice   = view(S.cache_Γpp, Ω, ν, :)
        F0slice  = view(S.cache_F0p, Ω, :, νp)

        # additional minus sign because we use crossing symmetry here
        for i in eachindex(Γslice)
            ω = value(meshes(S.cache_Γpp, Val(3))[i])
            Π0 = S.Π0pp[Ω, ω]

            val -= Γslice[i] * Π0 * F0slice[i]
        end

        return temperature(S) * val
    end

    # compute K3
    S.SGppL[3](S.FL.γp.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end

function BSE_K3!(
    S :: ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Γslice   = view(S.cache_Γpx, Ω, :, νp)
        Fslice   = view(S.cache_Fp,  Ω, ν,  :)
        F0slice  = view(S.cache_F0p, Ω, :, νp)

        for i in eachindex(Fslice)
            ω  = value(meshes(S.cache_Fp, Val(3))[i])
            Π0 = S.Π0pp[Ω, ω]
            Π  = S.Πpp[ Ω, ω]

            # 1ℓ and right part, additional minus sign because we use crossing symmetry here
            val -= Fslice[i] * ((Π - Π0) * F0slice[i] + Π * Γslice[i])

            # central part
            if is_inbounds(Ω - ω, meshes(S.FL.γp.K3, Val(2)))
                val += Fslice[i] * Π * S.FL.γp.K3[Ω, Ω - ω, νp]
            elseif is_inbounds(Ω - ω, meshes(S.FL.γp.K2, Val(2)))
                val += Fslice[i] * Π * S.FL.γp.K2[Ω, Ω - ω]
            end
        end

        return S.FL.γp.K3[Ω, ν, νp] + temperature(S) * val
    end

    # compute K3
    S.SGpp[3](S.Fbuff.γp.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end



function BSE_K3_mfRG!(
    S :: ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Γslice   = view(S.cache_Γpx, Ω, :, νp)
        Fslice   = view(S.cache_Fp,  Ω, ν,  :)
        F0slice  = view(S.cache_F0p, Ω, :, νp)

        for i in eachindex(Fslice)
            ω  = value(meshes(S.cache_Fp, Val(3))[i])
            Π0 = S.Π0pp[Ω, ω]

            # 1ℓ and right part, additional minus sign because we use crossing symmetry here
            val -= Fslice[i] * Π0 * Γslice[i]

            # central part
            if is_inbounds(Ω - ω, meshes(S.FL.γp.K3, Val(2)))
                val += Fslice[i] * Π0 * S.FL.γp.K3[Ω, Ω - ω, νp]
            elseif is_inbounds(Ω - ω, meshes(S.FL.γp.K2, Val(2)))
                val += Fslice[i] * Π0 * S.FL.γp.K2[Ω, Ω - ω]
            end
        end

        return S.FL.γp.K3[Ω, ν, νp] + temperature(S) * val
    end

    # compute K3
    S.SGpp[3](S.Fbuff.γp.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end
