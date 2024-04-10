function BSE_L_K3!(
    S :: ParquetSolver{Q},
    Γ :: MF_K3{Q},
    F0 :: MF_K3{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Π0slice  = view(S.Π0pp, Ω, :)
        Γslice   = view(Γ, Ω, ν, :)
        F0slice  = view(F0, Ω, :, νp)

        # additional minus sign because we use crossing symmetry here
        for i in 1 : length(meshes(Γ, 3))
            val -= Γslice[i] * Π0slice[i] * F0slice[i]
        end

        return temperature(S) * val
    end

    # compute K3
    S.SGppL[3](S.FL.γp.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end

function BSE_K3!(
    S :: ParquetSolver{Q},
    Γ :: MF_K3{Q},
    F :: MF_K3{Q},
    F0 :: MF_K3{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val      = zero(Q)
        Π0slice  = view(S.Π0pp, Ω, :)
        Πslice   = view(S.Πpp, Ω, :)
        Γslice   = view(Γ, Ω, :, νp)
        Fslice   = view(F, Ω, ν, :)
        F0slice  = view(F0, Ω, :, νp)

        # vectorize 1ℓ and right part, additional minus sign because we use crossing symmetry here
        for i in 1 : length(meshes(Γ, 2))
            val -= Fslice[i] * ((Πslice[i] - Π0slice[i]) * F0slice[i] + Πslice[i] * Γslice[i])
        end

        for i in eachindex(meshes(Γ, 2))
            ω  = value(meshes(Γ, 2)[i])

            # central part
            if is_inbounds(Ω - ω, meshes(S.FL.γp.K3, 2)) && is_inbounds(νp, meshes(S.FL.γp.K3, 3))
                val += Fslice[i] * Πslice[i] * S.FL.γp.K3[Ω, Ω - ω, νp]
            else
                val += Fslice[i] * Πslice[i] * S.FL.γp.K2(Ω, Ω - ω)
            end
        end

        return S.FL.γp.K3[Ω, ν, νp] + temperature(S) * val
    end

    # compute K3
    S.SGpp[3](S.Fbuff.γp.K3, InitFunction{3, Q}(diagram); mode = S.mode)

    return nothing
end
