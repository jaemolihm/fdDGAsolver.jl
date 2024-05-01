# Note: K1 contributions to left and right part always vanish
function BSE_K1!(
    K1   :: MF_K1{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
    SG   :: SymmetryGroup{1, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # model the diagram
    @inline function diagram(wtpl)

        Ω       = wtpl[1]
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)
        Πslice  = view(Π, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            if is_mfRG == Val(true)
                # vertices
                F0l = F0(Ω, νInf, ω, aCh, pSp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, aCh, pSp)

                # central part
                val += F0l * Π0slice[i] * FLr
            else
                # vertices
                Fl  = F( Ω, νInf, ω, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                # 1ℓ and central part
                val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
            end
        end

        return temperature(F) * val * sign
    end

    # compute K1
    SG(K1, InitFunction{1, Q}(diagram); mode)

    return nothing
end
