function BSE_L_K3!(
    S  :: ParquetSolver{Q},
       :: Type{tCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Γslice  = view(S.cache_Γt,  Ω, ν,  :)
        F0slice = view(S.cache_F0t, Ω, :, νp)

        # additional minus sign for xSp terms because we use crossing symmetry here
        for i in eachindex(Γslice)
            ω = value(meshes(S.cache_Γt, Val(3))[i])
            Π0 = S.Π0ph[Ω, ω]

            val -= Π0 * Γslice[i] * F0slice[i]
        end

        return temperature(S) * val
    end

    # compute K3
    S.SGphL[3](S.FL.γt.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    # Currently S.FL.γt.K3 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.FL.γt.K3, S.FL.γa.K3)
    S.FL.γt.K3.data ./= 2

    return nothing
end

function BSE_K3!(
    S  :: ParquetSolver{Q},
       :: Type{tCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Γslice  = view(S.cache_Γt,  Ω, νp, :)
        Fslice  = view(S.cache_Ft,  Ω, ν, :)
        F0slice = view(S.cache_F0t, Ω, :, νp)

        for i in eachindex(meshes(S.cache_Γt, Val(3)))
            ω = value(meshes(S.cache_Γt, Val(3))[i])
            Π0 = S.Π0ph[Ω, ω]
            Π  = S.Πph[ Ω, ω]

            # 1ℓ and right part, additional minus sign for xSp terms because we use crossing symmetry here
            val -= (Π - Π0) * Fslice[i] * F0slice[i] + Π * Fslice[i] * Γslice[i]

            # central part, additional minus sign for xSp terms because we use crossing symmetry here
            if is_inbounds(ω, meshes(S.FL.γt.K3, Val(2)))
                val -= Π * Fslice[i] * (2 * S.FL.γt.K3[Ω, ω, νp] - S.FL.γa.K3[Ω, ω, νp])
            elseif is_inbounds(ω, meshes(S.FL.γt.K2, Val(2)))
                val -= Π * Fslice[i] * (2 * S.FL.γt.K2[Ω, ω] - S.FL.γa.K2[Ω, ω])
            end
        end

        return 2 * S.FL.γt.K3[Ω, ν, νp] - S.FL.γa.K3[Ω, ν, νp] + temperature(S) * val
    end

    # compute K3
    S.SGph[3](S.Fbuff.γt.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    # Currently S.Fbuff.γt.K3 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K3, S.Fbuff.γa.K3)
    S.Fbuff.γt.K3.data ./= 2

    return nothing
end
