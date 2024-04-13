function BSE_L_K3!(
    S  :: NL2_ParquetSolver{Q},
    Γ  :: NL2_MF_K3{Q},
    F0 :: NL2_MF_K3{Q},
       :: Type{pCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(Γ,  Ω, ν,  :, P, :)
        F0slice = view(F0, Ω, :, νp, P, :)

        # additional minus sign because we use crossing symmetry here
        for i in eachindex(Γslice)
            ω = value(meshes(Γ, Val(3))[i.I[1]])
            q = value(meshes(Γ, Val(5))[i.I[2]])
            Π0 = S.Π0pp[Ω, ω, P, q]

            val -= Γslice[i] * Π0 * F0slice[i]
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K3
    S.SGppL[3](S.FL.γp.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end


function BSE_K3!(
    S  :: NL2_ParquetSolver{Q},
    Γ  :: NL2_MF_K3{Q},
    F  :: NL2_MF_K3{Q},
    F0 :: NL2_MF_K3{Q},
       :: Type{pCh}
    )  :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(Γ,  Ω, :, νp, P, :)
        Fslice  = view(F,  Ω, ν,  :, P, :)
        F0slice = view(F0, Ω, :, νp, P, :)

        for i in eachindex(Fslice)
            ω = value(meshes(F, Val(3))[i.I[1]])
            q = value(meshes(F, Val(5))[i.I[2]])
            Π0 = S.Π0pp[Ω, ω, P, q]
            Π  = S.Πpp[ Ω, ω, P, q]

            # 1ℓ and right part, additional minus sign because we use crossing symmetry here
            val -= Fslice[i] * ((Π - Π0) * F0slice[i] + Π * Γslice[i])

            # central part
            if is_inbounds(Ω - ω, meshes(S.FL.γp.K3, Val(2)))
                val += Fslice[i] * Π * S.FL.γp.K3[Ω, Ω - ω, νp, P]
            elseif is_inbounds(Ω - ω, meshes(S.FL.γp.K2, Val(2)))
                val += Fslice[i] * Π * S.FL.γp.K2[Ω, Ω - ω, P, fold_back(P - q)]
            end
        end

        return S.FL.γp.K3[Ω, ν, νp, P] + temperature(S) * val / numP_Γ(S)
    end

    # compute K3
    S.SGpp[3](S.Fbuff.γp.K3, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end
