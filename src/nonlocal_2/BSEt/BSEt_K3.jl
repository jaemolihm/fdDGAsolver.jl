function BSE_K3!(
    S  :: NL2_ParquetSolver{Q},
    Γ  :: NL2_MF_K3{Q},
    F  :: NL2_MF_K3{Q},
    F0 :: NL2_MF_K3{Q},
       :: Type{tCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(Γ,  Ω, νp,  :, P, :)  # using symmetry Γ[Ω, ω, νp] = Γ[Ω, νp, ω]
        Fslice  = view(F,  Ω,  ν,  :, P, :)
        F0slice = view(F0, Ω,  :, νp, P, :)

        for i in eachindex(Fslice)
            ω = value(meshes(F, Val(3))[i.I[1]])
            q = value(meshes(F, Val(5))[i.I[2]])
            Π0 = S.Π0ph[Ω, ω, P, q]
            Π  = S.Πph[ Ω, ω, P, q]

            # 1ℓ and right part, additional minus sign for xSp terms because we use crossing symmetry here
            val -= (Π - Π0) * Fslice[i] * F0slice[i] + Π * Fslice[i] * Γslice[i]

            # central part
            if is_inbounds(ω, meshes(S.FL.γt.K3, Val(2)))
                val -= Π * Fslice[i] * (2 * S.FL.γt.K3[Ω, ω, νp, P] - S.FL.γa.K3[Ω, ω, νp, P])
            elseif is_inbounds(ω, meshes(S.FL.γp.K2, Val(2)))
                val -= Π * Fslice[i] * (2 * S.FL.γt.K2[Ω, ω, P, q] - S.FL.γa.K2[Ω, ω, P, q])
            end
        end

        return 2 * S.FL.γt.K3[Ω, ν, νp, P] - S.FL.γa.K3[Ω, ν, νp, P] + temperature(S) * val / numP_Γ(S)
    end

    # compute K3
    S.SGph[3](S.Fbuff.γt.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    # Currently S.Fbuff.γt.K3 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K3, S.Fbuff.γa.K3)
    S.Fbuff.γt.K3.data ./= 2

    return nothing
end
