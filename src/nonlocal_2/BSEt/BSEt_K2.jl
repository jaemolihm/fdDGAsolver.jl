function BSE_L_K2!(
    S :: NL2_ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P, :)

        for iq in axes(Π0slice, 2)
            q = value(meshes(S.Π0ph, Val(4))[iq])
            Fview  = fixed_momentum_view(S.F,  P, k,  q, tCh)
            F0view = fixed_momentum_view(S.F0, P, q, k0, tCh)

            for iω in axes(Π0slice, 1)
                ω = value(meshes(S.Π0ph, Val(2))[iω])

                # vertices
                Γd  = Fview( Ω, ν,    ω, tCh, dSp; F0 = false, γt = false)
                F0d = F0view(Ω, ω, νInf, tCh, dSp)

                val -= Γd * Π0slice[iω, iq] * F0d
            end
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGphL[2](S.FL.γt.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    # Currently S.FL.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.FL.γt.K2, S.FL.γa.K2)
    S.FL.γt.K2.data ./= 2

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    S :: NL2_ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P, :)
        Πslice  = view(S.Πph , Ω, :, P, :)

        for iq in axes(Π0slice, 2)
            q = value(meshes(S.Π0ph, Val(4))[iq])
            Fview  = fixed_momentum_view(S.F,  P, k,  q, tCh)
            F0view = fixed_momentum_view(S.F0, P, q, k0, tCh)
            FLview = fixed_momentum_view(S.FL, P, q, k0, tCh)

            for iω in axes(Π0slice, 1)
                ω = value(meshes(S.Π0ph, Val(2))[iω])

                # vertices
                Fl  = Fview( Ω, ν,    ω, tCh, dSp) - Fview(Ω, νInf, ω, tCh, dSp)
                F0r = F0view(Ω, ω, νInf, tCh, dSp)
                FLr = FLview(Ω, ω, νInf, tCh, dSp)

                # 1ℓ and central part
                val -= Fl * ((Πslice[iω, iq] - Π0slice[iω, iq]) * F0r + Πslice[iω, iq] * FLr)
            end
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGph[2](S.Fbuff.γt.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    mult_add!(S.Fbuff.γt.K2, S.FL.γt.K2, 2)
    mult_add!(S.Fbuff.γt.K2, S.FL.γa.K2, -1)

    # Currently S.Fbuff.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K2, S.Fbuff.γa.K2)
    S.Fbuff.γt.K2.data ./= 2

    return nothing
end




# Note: K2 contributions to right part always vanishes
function BSE_K2_mfRG!(
    S :: NL2_ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(S.Π0ph, Val(2)), meshes(S.Π0ph, Val(4))), view(S.Π0ph, Ω, :, P, :))

        for iq in eachindex(meshes(S.Fbuff.γa.K2, Val(4)))
            q = value(meshes(S.Fbuff.γa.K2, Val(4))[iq])
            F0view = fixed_momentum_view(S.F0, P, k,  q, tCh)
            FLview = fixed_momentum_view(S.FL, P, q, k0, tCh)

            for iω in eachindex(meshes(S.Fbuff.γp.K2, Val(2)))
                ω = value(meshes(S.Fbuff.γp.K2, Val(2))[iω])

                # vertices
                Fl  = F0view(Ω, ν,    ω, tCh, dSp) - F0view(Ω, νInf, ω, tCh, dSp)
                FLr = FLview(Ω, ω, νInf, tCh, dSp)

                # 1ℓ and central part
                val -= Fl * Π0slice(ω, q) * FLr
            end
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGph[2](S.Fbuff.γt.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    mult_add!(S.Fbuff.γt.K2, S.FL.γt.K2, 2)
    mult_add!(S.Fbuff.γt.K2, S.FL.γa.K2, -1)

    # Currently S.Fbuff.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K2, S.Fbuff.γa.K2)
    S.Fbuff.γt.K2.data ./= 2

    return nothing
end
