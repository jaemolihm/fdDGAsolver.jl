function BSE_L_K3!(
    S :: ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Γslice   = view(S.cache_Γa, Ω, ν, :)
        F0slice  = view(S.cache_F0a, Ω, :, νp)

        for i in eachindex(Γslice)
            ω = value(meshes(S.cache_Γa, Val(3))[i])
            Π0 = S.Π0ph[Ω, ω]

            val += Γslice[i] * Π0 * F0slice[i]
        end

        return temperature(S) * val
    end

    # compute K3
    S.SGphL[3](S.FL.γa.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end

function BSE_K3!(
    S :: ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Γslice   = view(S.cache_Γa,  Ω, νp, :)
        Fslice   = view(S.cache_Fa,  Ω, ν, :)
        F0slice  = view(S.cache_F0a, Ω, :, νp)

        for i in eachindex(Fslice)
            ω = value(meshes(S.cache_Fa, Val(3))[i])
            Π0 = S.Π0ph[Ω, ω]
            Π  = S.Πph[ Ω, ω]

            # 1ℓ and right part
            val += Fslice[i] * ((Π - Π0) * F0slice[i] + Π * Γslice[i])

            # central part
            if is_inbounds(ω, meshes(S.FL.γa.K3, Val(2)))
                val += Fslice[i] * Π * S.FL.γa.K3[Ω, ω, νp]
            end
        end

        return S.FL.γa.K3[Ω, ν, νp] + temperature(S) * val
    end

    # compute K3
    S.SGph[3](S.Fbuff.γa.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end



function BSE_K3_mfRG!(
    S :: ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Γslice   = view(S.cache_Γa,  Ω, νp, :)
        Fslice   = view(S.cache_Fa,  Ω, ν, :)
        F0slice  = view(S.cache_F0a, Ω, :, νp)

        for i in eachindex(Fslice)
            ω = value(meshes(S.cache_Fa, Val(3))[i])
            Π0 = S.Π0ph[Ω, ω]
            Π  = S.Πph[ Ω, ω]

            # 1ℓ and right part
            val += Fslice[i] * Π0 * Γslice[i]

            # central part
            if is_inbounds(ω, meshes(S.FL.γa.K3, Val(2)))
                val += Fslice[i] * Π0 * S.FL.γa.K3[Ω, ω, νp]
            end
        end

        return S.FL.γa.K3[Ω, ν, νp] + temperature(S) * val
    end

    # compute K3
    S.SGph[3](S.Fbuff.γa.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end
