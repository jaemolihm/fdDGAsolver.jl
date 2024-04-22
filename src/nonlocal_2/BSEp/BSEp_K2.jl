function BSE_L_K2!(
    S :: NL2_ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(S.Π0pp, Val(2)), meshes(S.Π0pp, Val(4))), view(S.Π0pp, Ω, :, P, :))

        for iq in eachindex(meshes(S.FL.γp.K2, Val(4)))
            q = value(meshes(S.FL.γp.K2, Val(4))[iq])

            Fview  = fixed_momentum_view(S.F,  P,     k,  q, pCh)
            F0view = fixed_momentum_view(S.F0, P, fold_back(P - q, meshes(S.FL.γp.K2, Val(4))), k0, pCh)

            for iω in eachindex(meshes(S.FL.γp.K2, Val(2)))
                ω = value(meshes(S.FL.γp.K2, Val(2))[iω])

                # Vertices
                Γp  = Fview( Ω,     ν,    ω, pCh, pSp; F0 = false, γp = false)
                F0p = F0view(Ω, Ω - ω, νInf, pCh, pSp)

                val += Γp * Π0slice[ω, q] * F0p
            end
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGppL[2](S.FL.γp.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    S :: NL2_ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(S.Π0pp, Ω, :, P, :)
        Πslice  = view(S.Πpp , Ω, :, P, :)

        for iq in axes(Π0slice, 2)
            q = value(meshes(S.Π0pp, Val(4))[iq])

            Fview  = fixed_momentum_view(S.F,  P,     k,  q, pCh)
            F0view = fixed_momentum_view(S.F0, P, fold_back(P - q, meshes(S.FL.γp.K2, Val(4))), k0, pCh)
            FLview = fixed_momentum_view(S.FL, P, fold_back(P - q, meshes(S.FL.γp.K2, Val(4))), k0, pCh)

            for iω in axes(Π0slice, 1)
                ω = value(meshes(S.Π0pp, Val(2))[iω])

                # vertices
                Fl  = Fview( Ω,     ν,    ω, pCh, pSp) - Fview(Ω, νInf, ω, pCh, pSp)
                F0r = F0view(Ω, Ω - ω, νInf, pCh, pSp)
                FLr = FLview(Ω, Ω - ω, νInf, pCh, pSp)

                # 1ℓ and central part
                val += Fl * ((Πslice[iω, iq] - Π0slice[iω, iq]) * F0r + Πslice[iω, iq] * FLr)
            end
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGpp[2](S.Fbuff.γp.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γp.K2, S.FL.γp.K2)

    return nothing
end



function BSE_K2_mfRG!(
    S :: NL2_ParquetSolver{Q},
      :: Type{pCh}
    ) :: Nothing where {Q}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(S.Π0pp, Val(2)), meshes(S.Π0pp, Val(4))), view(S.Π0pp, Ω, :, P, :))

        for iq in eachindex(meshes(S.Fbuff.γp.K2, Val(4)))
            q = value(meshes(S.Fbuff.γp.K2, Val(4))[iq])
            F0view = fixed_momentum_view(S.F0, P,     k,  q, pCh)
            FLview = fixed_momentum_view(S.FL, P, fold_back(P - q, meshes(S.FL.γp.K2, Val(4))), k0, pCh)

            for iω in eachindex(meshes(S.Fbuff.γp.K2, Val(2)))
                ω = value(meshes(S.Fbuff.γp.K2, Val(2))[iω])

                # vertices
                Fl  = F0view(Ω, ν, Ω - ω, pCh, pSp) - F0view(Ω, νInf, Ω - ω, pCh, pSp)
                FLr = FLview(Ω, ω,  νInf, pCh, pSp)

                # 1ℓ and central part
                val += Fl * Π0slice(Ω - ω, q) * FLr
            end
        end

        return temperature(S) * val / numP_Γ(S)
    end

    # compute K2
    S.SGpp[2](S.Fbuff.γp.K2, InitFunction{4, Q}(diagram); mode = S.mode)

    add!(S.Fbuff.γp.K2, S.FL.γp.K2)

    return nothing
end
