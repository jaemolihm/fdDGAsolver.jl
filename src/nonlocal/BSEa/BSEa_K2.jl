function BSE_L_K2!(
    K2   :: NL_MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    Π0   :: NL_MF_Π{Q},
    Π    :: NL_MF_Π{Q},
    SG   :: SymmetryGroup{3, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(Π0, Val(2)),), view(Π0, Ω, :, P))

        for iω in eachindex(meshes(K2, Val(2)))
            ω = value(meshes(K2, Val(2))[iω])

            Γl  = F( Ω, ν, _crossing(Ω, ω, Ch), P, kSW, kSW, Ch, Sp; F0 = false, γa = (Ch !== aCh), γp = (Ch !== pCh), γt = (Ch !== tCh))
            F0r = F0(Ω, ω, νInf, P, kSW, kSW, Ch, Sp)

            val += Γl * Π0slice[ω] * F0r
        end

        return temperature(F) * val * sign
    end

    # compute K2
    SG(K2, InitFunction{3, Q}(diagram); mode)

    return nothing
end


# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    K2   :: NL_MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: NL_MF_Π{Q},
    Π    :: NL_MF_Π{Q},
    SG   :: SymmetryGroup{3, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P)
        Πslice  = view(Π , Ω, :, P)

        for iω in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[iω])

            if is_mfRG === Val(true)

                Fl  = F0(Ω, ν, _crossing(Ω, ω, Ch), P, kSW, kSW, Ch, Sp) - F0(Ω, νInf, _crossing(Ω, ω, Ch), P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, ω, νInf, P, kSW, kSW, Ch, Sp)

                # 1ℓ and central part
                val += Fl * Π0slice[iω] * FLr

            else

                Fl  = F( Ω, ν, ω, P, kSW, kSW, Ch, Sp) - F( Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)

                # 1ℓ and central part
                val += Fl * ((Πslice[iω] - Π0slice[iω]) * F0r + Πslice[iω] * FLr)

            end

        end

        return temperature(F) * val * sign
    end

    # compute K2
    SG(K2, InitFunction{3, Q}(diagram); mode)

    if Ch === tCh
        mult_add!(K2, FL.γt.K2, 2)
        mult_add!(K2, FL.γa.K2, -1)
    else
        add!(K2, get_reducible_vertex(FL, Ch).K2)
    end

    return nothing
end


function BSE_K2_new!(
    K2   :: NL_MF_K2{Q},
    F0   :: AbstractVertex{Q},
    F    :: AbstractVertex{Q},
    Π0   :: NL_MF_Π{Q},
    Π    :: NL_MF_Π{Q},
    SG   :: SymmetryGroup{3, Q},
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

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P)
        Πslice  = view(Π , Ω, :, P)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            # vertices
            Fl  = F( Ω, ν, ω, P, kSW, kSW, Ch, Sp) - F( Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
            F0l = F0(Ω, ν, ω, P, kSW, kSW, Ch, Sp) - F0(Ω, νInf, ω, P, kSW, kSW, Ch, Sp)

            # 1ℓ and central part
            if is_mfRG === Val(true)
                val += (Fl - F0l) * Πslice[i] * U
            else
                val += (Fl * Πslice[i] - F0l * Π0slice[i]) * U
            end
        end

        return temperature(F) * val * sign
    end

    # compute K2
    SG(K2, InitFunction{3, Q}(diagram); mode)

    return nothing
end
