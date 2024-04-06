function BSE_L_K3!(
    S :: NL_ParquetSolver{Q},
    Γ :: NL_MF_K3{Q},
    F0 :: NL_MF_K3{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(Γ,  Ω, ν,  :, P)
        F0slice = view(F0, Ω, :, νp, P)

        # additional minus sign because we use crossing symmetry here
        for i in eachindex(meshes(Γ, 3))
            ω = value(meshes(Γ, 3)[i])
            Π0 = S.Π0pp[Ω, ω, P, kSW]

            val -= Γslice[i] * Π0 * F0slice[i]
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K3
    S.SGppL[3](S.FL.γp.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end

function BSE_K3!(
    S :: NL_ParquetSolver{Q},
    Γ :: NL_MF_K3{Q},
    F :: NL_MF_K3{Q},
    F0 :: NL_MF_K3{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(Γ,  Ω, :, νp, P)
        Fslice  = view(F,  Ω, ν,  :, P)
        F0slice = view(F0, Ω, :, νp, P)

        for i in eachindex(meshes(Γ, 2))
            ω  = value(meshes(Γ, 2)[i])
            Π0 = S.Π0pp[Ω, ω, P, kSW]
            Π  = S.Πpp[ Ω, ω, P, kSW]

            # 1ℓ and right part, additional minus sign because we use crossing symmetry here
            val -= Fslice[i] * ((Π - Π0) * F0slice[i] + Π * Γslice[i])

            # central part
            if is_inbounds(Ω - ω, meshes(S.FL.γp.K3, 2)) && is_inbounds(νp, meshes(S.FL.γp.K3, 3))
                val += Fslice[i] * Π * S.FL.γp.K3[Ω, Ω - ω, νp, P]
            else
                val += Fslice[i] * Π * box_eval(S.FL.γp.K2, Ω, Ω - ω, P)
            end
        end

        return S.FL.γp.K3[Ω, ν, νp, P] + temperature(S) * val / numP_Γ(S)
    end

    # compute K3
    S.SGpp[3](S.Fbuff.γp.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end
