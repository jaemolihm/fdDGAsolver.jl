function BSE_L_K3!(
    S :: ParquetSolver,
    Γ :: MF_K3{Q},
    F0 :: MF_K3{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Π0slice  = view(S.Π0ph, Ω, :)
        Γslice   = view(Γ, Ω, ν, :)
        F0slice  = view(F0, Ω, :, νp)

        for i in 1 : length(meshes(Γ, 3))
            val += Γslice[i] * Π0slice[i] * F0slice[i]
        end

        return temperature(S) * val
    end

    # compute K3
    S.SGphL[3](S.FL.γa.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end

function BSE_K3!(
    S :: ParquetSolver,
    Γ :: MF_K3{Q},
    F :: MF_K3{Q},
    F0 :: MF_K3{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Π0slice  = view(S.Π0ph, Ω, :)
        Πslice   = view(S.Πph, Ω, :)
        Γslice   = view(Γ, Ω, νp, :)
        Fslice   = view(F, Ω, ν, :)
        F0slice  = view(F0, Ω, :, νp)

        # vectorize 1ℓ and right part
        for i in 1 : length(meshes(Γ, 3))
            val += Fslice[i] * ((Πslice[i] - Π0slice[i]) * F0slice[i] + Πslice[i] * Γslice[i])
        end

        for i in eachindex(meshes(Γ, 3))
            ω = value(meshes(Γ, 3)[i])

            # central part
            if is_inbounds(ω, meshes(S.FL.γa.K3, 2)) && is_inbounds(νp, meshes(S.FL.γa.K3, 3))
                val += Fslice[i] * Πslice[i] * S.FL.γa.K3[Ω, ω, νp]
            else
                val += Fslice[i] * Πslice[i] * S.FL.γa.K2(Ω, ω)
            end
        end

        return S.FL.γa.K3[Ω, ν, νp] + temperature(S) * val
    end

    # compute K3
    S.SGph[3](S.Fbuff.γa.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end
