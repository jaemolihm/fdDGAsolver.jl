# Note: K1 contributions to left and right part always vanish
function BSE_K1!(
    K1   :: NL_MF_K1{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: NL_MF_Π{Q},
    Π    :: NL_MF_Π{Q},
    SG   :: SymmetryGroup{2, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P)
        Πslice  = view(Π , Ω, :, P)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            if is_mfRG === Val(true)
                # mfRG for the K1 class
                # vertices
                Fl  = F0(Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)

                # central part
                val += Fl * Π0slice[i] * FLr

            else
                # fdDΓA for the K1 class
                # vertices
                Fl  = F( Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)

                # 1ℓ and central part
                val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
            end
        end

        return temperature(F) * val * sign
    end

    # compute K1
    SG(K1, InitFunction{2, Q}(diagram); mode)

    return nothing
end
