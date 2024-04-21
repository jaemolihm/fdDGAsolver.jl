function BSE_L_K2!(
    S :: NL_ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(S.Π0ph, Val(2)),), view(S.Π0ph, Ω, :, P))

        for iω in eachindex(meshes(S.FL.γt.K2, Val(2)))
            ω = value(meshes(S.FL.γt.K2, Val(2))[iω])

            # vertices
            Γd  = S.F( Ω, ν,    ω, P, kSW, kSW, tCh, dSp; F0 = false, γt = false)
            F0d = S.F0(Ω, ω, νInf, P, kSW, kSW, tCh, dSp)

            val -= Γd * Π0slice[ω] * F0d
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGphL[2](S.FL.γt.K2, InitFunction{3, Q}(diagram); mode = S.mode)

    # Currently S.FL.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.FL.γt.K2, S.FL.γa.K2)
    S.FL.γt.K2.data ./= 2

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    S :: NL_ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P)
        Πslice  = view(S.Πph , Ω, :, P)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, Val(2))[i])

            # vertices
            Fl  = S.F( Ω, ν,    ω, P, kSW, kSW, tCh, dSp) - S.F(Ω, νInf, ω, P, kSW, kSW, tCh, dSp)
            F0r = S.F0(Ω, ω, νInf, P, kSW, kSW, tCh, dSp)
            FLr = S.FL(Ω, ω, νInf, P, kSW, kSW, tCh, dSp)

            # 1ℓ and central part
            val -= Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGph[2](S.Fbuff.γt.K2, InitFunction{3, Q}(diagram); mode = S.mode)

    mult_add!(S.Fbuff.γt.K2, S.FL.γt.K2, 2)
    mult_add!(S.Fbuff.γt.K2, S.FL.γa.K2, -1)

    # Currently S.Fbuff.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K2, S.Fbuff.γa.K2)
    S.Fbuff.γt.K2.data ./= 2

    return nothing
end



function BSE_K2_mfRG!(
    S :: NL_ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(S.Π0ph, Val(2)),), view(S.Π0ph, Ω, :, P))

        for iω in eachindex(meshes(S.FL.γt.K2, Val(2)))
            ω = value(meshes(S.FL.γt.K2, Val(2))[iω])

            # vertices
            Fl  = S.F0(Ω, ν,    ω, P, kSW, kSW, tCh, dSp) - S.F0(Ω, νInf, ω, P, kSW, kSW, tCh, dSp)
            FLr = S.FL(Ω, ω, νInf, P, kSW, kSW, tCh, dSp)

            # 1ℓ and central part
            val -= Fl * Π0slice[ω] * FLr
        end

        return temperature(S) * val
    end

    # compute K2
    S.SGph[2](S.Fbuff.γt.K2, InitFunction{3, Q}(diagram); mode = S.mode)

    mult_add!(S.Fbuff.γt.K2, S.FL.γt.K2, 2)
    mult_add!(S.Fbuff.γt.K2, S.FL.γa.K2, -1)

    # Currently S.Fbuff.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K2, S.Fbuff.γa.K2)
    S.Fbuff.γt.K2.data ./= 2

    return nothing
end
