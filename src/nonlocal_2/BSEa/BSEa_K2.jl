function BSE_L_K2!(
    K2   :: NL2_MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
    SG   :: SymmetryGroup{4, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(Π0, Val(2)), meshes(Π0, Val(4))), view(Π0, Ω, :, P, :))

        for iq in eachindex(meshes(K2, Val(4)))
            q = value(meshes(K2, Val(4))[iq])
            # Fview  = fixed_momentum_view(F,  P, k, _crossing(P, q, Ch), Ch)
            # F0view = fixed_momentum_view(F0, P, q, k0, Ch)

            for iω in eachindex(meshes(K2, Val(2)))
                ω = value(meshes(K2, Val(2))[iω])

                # vertices
                # Γl  = Fview( Ω, ν, _crossing(Ω, ω, Ch), Ch, Sp; F0 = false, γa = (Ch !== aCh), γp = (Ch !== pCh), γt = (Ch !== tCh))
                # F0r = F0view(Ω, ω, νInf, Ch, Sp)

                Γl  = F( Ω, ν, _crossing(Ω, ω, Ch), P, k, _crossing(P, q, Ch), Ch, Sp; F0 = false, γa = (Ch !== aCh), γp = (Ch !== pCh), γt = (Ch !== tCh))
                F0r = F0(Ω, ω, νInf, P, q, k0, Ch, Sp)

                val += Γl * Π0slice[ω, q] * F0r
            end
        end

        return temperature(F) * val / numP(F) * sign
    end

    # compute K2
    SG(K2, InitFunction{4, Q}(diagram); mode)

    return nothing
end




# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    K2   :: NL2_MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
    SG   :: SymmetryGroup{4, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Val{false},
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P, :)
        Πslice  = view(Π , Ω, :, P, :)

        for iq in axes(Π0slice, 2)
            q = value(meshes(Π0, Val(4))[iq])

            # if is_mfRG === Val(true)
            #     F0view = fixed_momentum_view(F0, P, k,  q, Ch)
            #     FLview = fixed_momentum_view(FL, P, _crossing(P, q, Ch), k0, Ch)
            # else
            #     Fview  = fixed_momentum_view(F,  P, k,  q, Ch)
            #     F0view = fixed_momentum_view(F0, P, _crossing(P, q, Ch), k0, Ch)
            #     FLview = fixed_momentum_view(FL, P, _crossing(P, q, Ch), k0, Ch)
            # end

            for iω in axes(Π0slice, 1)
                ω = value(meshes(Π0, Val(2))[iω])

                if is_mfRG === Val(true)

                    # vertices
                    # Fl  = F0view(Ω, ν, _crossing(Ω, ω, Ch), Ch, Sp) - F0view(Ω, νInf, _crossing(Ω, ω, Ch), Ch, Sp)
                    # FLr = FLview(Ω, ω, νInf, Ch, Sp)

                    Fl  = F0(Ω, ν, _crossing(Ω, ω, Ch), P, k, _crossing(P, q, Ch), Ch, Sp) - F0(Ω, νInf, _crossing(Ω, ω, Ch), P, k, _crossing(P, q, Ch), Ch, Sp)
                    FLr = FL(Ω, ω, νInf, P, q, k0, Ch, Sp)

                    # 1ℓ and central part
                    val += Fl * Π0slice[ω, q] * FLr

                else

                    # vertices
                    # Fl  = Fview( Ω, ν,    ω, Ch, Sp) - Fview( Ω, νInf, ω, Ch, Sp)
                    # F0r = F0view(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)
                    # FLr = FLview(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                    Fl  = F( Ω, ν, ω, P, k, q, Ch, Sp) - F( Ω, νInf, ω, P, k, q, Ch, Sp)
                    F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)
                    FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)

                    # 1ℓ and central part
                    val += Fl * ((Πslice[iω, iq] - Π0slice[iω, iq]) * F0r + Πslice[iω, iq] * FLr)

                end
            end
        end

        return temperature(F) * val / numP(F) * sign
    end

    # compute K2
    SG(K2, InitFunction{4, Q}(diagram); mode)

    if Ch === tCh
        mult_add!(K2, FL.γt.K2, 2)
        mult_add!(K2, FL.γa.K2, -1)
    else
        add!(K2, get_reducible_vertex(FL, Ch).K2)
    end

    return nothing
end



function BSE_K2_new!(
    K2   :: NL2_MF_K2{Q},
    F0   :: AbstractVertex{Q},
    F    :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
    SG   :: SymmetryGroup{4, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    U = bare_vertex(F, Sp)

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        # Π0slice = view(Π0, Ω, :, P, :)
        # Πslice  = view(Π , Ω, :, P, :)

        Π0slice = MeshFunction((meshes(Π0, Val(2)), meshes(Π0, Val(4))), view(Π0, Ω, :, P, :))
        Πslice  = MeshFunction((meshes(Π,  Val(2)), meshes(Π,  Val(4))), view(Π,  Ω, :, P, :))

        # for iq in axes(Π0slice, 2)
        #     q = value(meshes(Π0, Val(4))[iq])
        #     Fview  = fixed_momentum_view(F,  P, k, q, Ch)
        #     F0view = fixed_momentum_view(F0, P, k, q, Ch)

        #     for iω in axes(Π0slice, 1)
        #         ω = value(meshes(Π0, Val(2))[iω])

        #         # vertices
        #         Fl  = Fview( Ω, ν, ω, Ch, Sp) - Fview( Ω, νInf, ω, Ch, Sp)
        #         F0l = F0view(Ω, ν, ω, Ch, Sp) - F0view(Ω, νInf, ω, Ch, Sp)

        #         # 1ℓ and central part
        #         if is_mfRG === Val(true)
        #             val += (Fl - F0l) * Πslice[iω, iq] * U
        #         else
        #             val += (Fl * Πslice[iω, iq] - F0l * Π0slice[iω, iq]) * U
        #         end
        #     end
        # end

        # for i in eachindex(Π0slice)
        #     ω = value(meshes(Π0, Val(2))[i.I[1]])
        #     q = value(meshes(Π0, Val(4))[i.I[2]])

        #     # vertices
        #     Fl  = F( Ω, ν, ω, P, k, q, Ch, Sp) - F( Ω, νInf, ω, P, k, q, Ch, Sp)
        #     F0l = F0(Ω, ν, ω, P, k, q, Ch, Sp) - F0(Ω, νInf, ω, P, k, q, Ch, Sp)

        #     # 1ℓ and central part
        #     if is_mfRG === Val(true)
        #         val += (Fl - F0l) * Πslice[i] * U
        #     else
        #         val += (Fl * Πslice[i] - F0l * Π0slice[i]) * U
        #     end
        # end

        for iq in eachindex(meshes(K2, Val(4))), iω in eachindex(meshes(K2, Val(2)))
            ω = value(meshes(K2, Val(2))[iω])
            q = value(meshes(K2, Val(4))[iq])

            # vertices
            Fl  = F( Ω, ν, ω, P, k, q, Ch, Sp) - F( Ω, νInf, ω, P, k, q, Ch, Sp)
            F0l = F0(Ω, ν, ω, P, k, q, Ch, Sp) - F0(Ω, νInf, ω, P, k, q, Ch, Sp)

            # 1ℓ and central part
            if is_mfRG === Val(true)
                val += (Fl - F0l) * Πslice[ω, q] * U
            else
                val += (Fl * Πslice[ω, q] - F0l * Π0slice[ω, q]) * U
            end
        end

        return temperature(F) * val / numP(F) * sign
    end

    # compute K2
    SG(K2, InitFunction{4, Q}(diagram); mode)

    return nothing
end
