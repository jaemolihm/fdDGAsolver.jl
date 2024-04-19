function BSE_L_K2!(
    S :: NL2_ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P, :)

        for iq in axes(Π0slice, 2)
            q = value(meshes(S.Π0ph, Val(4))[iq])
            Fview  = fixed_momentum_view(S.F,  P, k,  q, aCh)
            F0view = fixed_momentum_view(S.F0, P, q, k0, aCh)

            for iω in axes(Π0slice, 1)
                ω = value(meshes(S.Π0ph, Val(2))[iω])

                # vertices
                Γp  = Fview( Ω, ν,    ω, aCh, pSp; F0 = false, γa = false)
                F0p = F0view(Ω, ω, νInf, aCh, pSp)

                val += Γp * Π0slice[iω, iq] * F0p
            end
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGphL[2](S.FL.γa.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    S :: NL2_ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P, :)
        Πslice  = view(S.Πph , Ω, :, P, :)

        for iq in axes(Π0slice, 2)
            q = value(meshes(S.Π0ph, Val(4))[iq])
            Fview  = fixed_momentum_view(S.F,  P, k,  q, aCh)
            F0view = fixed_momentum_view(S.F0, P, q, k0, aCh)
            FLview = fixed_momentum_view(S.FL, P, q, k0, aCh)

            for iω in axes(Π0slice, 1)
                ω = value(meshes(S.Π0ph, Val(2))[iω])

                # vertices
                Fl  = Fview( Ω, ν,    ω, aCh, pSp) - Fview( Ω, νInf, ω, aCh, pSp)
                F0r = F0view(Ω, ω, νInf, aCh, pSp)
                FLr = FLview(Ω, ω, νInf, aCh, pSp)

                # 1ℓ and central part
                val += Fl * ((Πslice[iω, iq] - Π0slice[iω, iq]) * F0r + Πslice[iω, iq] * FLr)
            end
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGph[2](S.Fbuff.γa.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γa.K2, S.FL.γa.K2)

    return nothing
end


# Note: K2 contributions to right part always vanishes
function BSE_K2_mfRG!(
    S :: NL2_ParquetSolver{Q},
      :: Type{aCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P, :)

        for iq in axes(Π0slice, 2)
            q = value(meshes(S.Π0ph, Val(4))[iq])
            F0view = fixed_momentum_view(S.F0, P, k,  q, aCh)
            FLview = fixed_momentum_view(S.FL, P, q, k0, aCh)

            for iω in axes(Π0slice, 1)
                ω = value(meshes(S.Π0ph, Val(2))[iω])

                # vertices
                Fl  = F0view(Ω, ν,    ω, aCh, pSp) - F0view(Ω, νInf, ω, aCh, pSp)
                FLr = FLview(Ω, ω, νInf, aCh, pSp)

                # 1ℓ and central part
                val += Fl * Π0slice[iω, iq] * FLr
            end
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGph[2](S.Fbuff.γa.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γa.K2, S.FL.γa.K2)

    return nothing
end
