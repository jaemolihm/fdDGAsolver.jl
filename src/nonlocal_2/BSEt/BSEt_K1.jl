# Note: K1 contributions to left and right part always vanish
function BSE_K1!(
    S :: NL2_ParquetSolver{Q},
      :: Type{tCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0ph, Ω, :, P, :)
        Πslice  = view(S.Πph , Ω, :, P, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(S.Π0ph, Val(2))[i.I[1]])
            q = value(meshes(S.Π0ph, Val(4))[i.I[2]])

            # vertices
            Fl  = S.F( Ω, νInf,    ω, P, k0,  q, tCh, dSp)
            F0r = S.F0(Ω,    ω, νInf, P,  q, k0, tCh, dSp)
            FLr = S.FL(Ω,    ω, νInf, P,  q, k0, tCh, dSp)

            # 1ℓ and central part
            val -= Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K1
    S.SGph[1](S.Fbuff.γt.K1, InitFunction{2, Q}(diagram); mode = S.mode)

    # Currently S.Fbuff.γt.K1 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K1, S.Fbuff.γa.K1)
    S.Fbuff.γt.K1.data ./= 2

    return nothing
end
