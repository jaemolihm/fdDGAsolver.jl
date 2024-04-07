function BSE_K3!(
    S  :: NL2_ParquetSolver{Q},
    Γ  :: NL_MF_K3{Q},
    F  :: NL_MF_K3{Q},
    F0 :: NL_MF_K3{Q},
       :: Type{tCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val      = zero(Q)
        Γslice  = view(Γ,  Ω, νp,  :, P)  # using symmetry Γ[Ω, ω, νp] = Γ[Ω, νp, ω]
        Fslice  = view(F,  Ω,  ν,  :, P)
        F0slice = view(F0, Ω,  :, νp, P)

        for i in eachindex(meshes(Γ, 3))
            ω = value(meshes(Γ, 3)[i])
            Π0 = S.Π0ph[Ω, ω, P, kSW]
            Π  = S.Πph[ Ω, ω, P, kSW]

            # 1ℓ and right part, additional minus sign for xSp terms because we use crossing symmetry here
            val -= (Π - Π0) * Fslice[i] * F0slice[i] + Π * Fslice[i] * Γslice[i]

            # central part
            if is_inbounds(ω, meshes(S.FL.γt.K3, 2)) && is_inbounds(νp, meshes(S.FL.γt.K3, 3))
                val -= Π * Fslice[i] * (2 * S.FL.γt.K3[Ω, ω, νp, P] - S.FL.γa.K3[Ω, ω, νp, P])
            else
                val -= Π * Fslice[i] * (2 * box_eval(S.FL.γt.K2, Ω, ω, P, kSW) - box_eval(S.FL.γa.K2, Ω, ω, P, kSW))
            end
        end

        return 2 * S.FL.γt.K3[Ω, ν, νp, P] - S.FL.γa.K3[Ω, ν, νp, P] + temperature(S) * val
    end

    # compute K3
    S.SGph[3](S.Fbuff.γt.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    # Currently S.Fbuff.γt.K3 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K3, S.Fbuff.γa.K3)
    S.Fbuff.γt.K3.data ./= 2

    return nothing
end
