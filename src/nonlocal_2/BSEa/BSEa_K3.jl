function BSE_L_K3!(
    S  :: NL2_ParquetSolver{Q},
    Γ  :: NL2_MF_K3{Q},
    F0 :: NL2_MF_K3{Q},
       :: Type{aCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(Γ,  Ω, ν,  :, P, :)
        F0slice = view(F0, Ω, :, νp, P, :)

        for i in eachindex(Γslice)
            ω = value(meshes(Γ, Val(3))[i.I[1]])
            q = value(meshes(Γ, Val(5))[i.I[2]])
            Π0 = S.Π0ph[Ω, ω, P, q]

            val += Γslice[i] * Π0 * F0slice[i]
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K3
    S.SGphL[3](S.FL.γa.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end


function BSE_K3!(
    S  :: NL2_ParquetSolver,
    Γ  :: NL2_MF_K3{Q},
    F  :: NL2_MF_K3{Q},
    F0 :: NL2_MF_K3{Q},
       :: Type{aCh}
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

            # 1ℓ and right part
            val += Fslice[i] * ((Π - Π0) * F0slice[i] + Π * Γslice[i])

            # central part
            if is_inbounds(ω, meshes(S.FL.γa.K3, Val(2)))
                val += Fslice[i] * Π * S.FL.γa.K3[Ω, ω, νp, P]
            elseif is_inbounds(ω, meshes(S.FL.γa.K2, Val(2)))
                val += Fslice[i] * Π * S.FL.γa.K2[Ω, ω, P, q]
            end
        end

        return S.FL.γa.K3[Ω, ν, νp, P] + temperature(S) * val / numP_Γ(S)
    end

    # compute K3
    S.SGph[3](S.Fbuff.γa.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end
