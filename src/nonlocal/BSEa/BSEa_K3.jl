function BSE_L_K3!(
    S :: Union{NL_ParquetSolver{Q}, NL2_ParquetSolver{Q}},
    Γ :: NL_MF_K3{Q},
    F0 :: NL_MF_K3{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(Γ,  Ω, ν,  :, P)
        F0slice = view(F0, Ω, :, νp, P)

        for i in eachindex(meshes(Γ, Val(3)))
            ω = value(meshes(Γ, Val(3))[i])
            Π0 = S.Π0ph[Ω, ω, P, kSW]

            val += Γslice[i] * Π0 * F0slice[i]
        end

        return temperature(S) * val
    end

    # compute K3
    S.SGphL[3](S.FL.γa.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end

function BSE_K3!(
    S :: NL_ParquetSolver,
    Γ :: NL_MF_K3{Q},
    F :: NL_MF_K3{Q},
    F0 :: NL_MF_K3{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(Γ, Ω, νp, :, P)
        Fslice  = view(F, Ω, ν, :, P)
        F0slice = view(F0, Ω, :, νp, P)

        for i in eachindex(meshes(Γ, Val(3)))
            ω = value(meshes(Γ, Val(3))[i])
            Π0 = S.Π0ph[Ω, ω, P, kSW]
            Π  = S.Πph[ Ω, ω, P, kSW]

            # 1ℓ and right part
            val += Fslice[i] * ((Π - Π0) * F0slice[i] + Π * Γslice[i])

            # central part
            if is_inbounds(ω, meshes(S.FL.γa.K3, Val(2)))
                val += Fslice[i] * Π * S.FL.γa.K3[Ω, ω, νp, P]
            elseif is_inbounds(ω, meshes(S.FL.γa.K2, Val(2)))
                val += Fslice[i] * Π * S.FL.γa.K2[Ω, ω, P]
            end
        end

        return S.FL.γa.K3[Ω, ν, νp, P] + temperature(S) * val
    end

    # compute K3
    S.SGph[3](S.Fbuff.γa.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end
