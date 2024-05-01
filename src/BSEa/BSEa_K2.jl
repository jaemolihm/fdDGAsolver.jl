function BSE_L_K2!(
    K2   :: MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
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

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            # vertices
            Γp  = F(Ω, ν, ω, Ch, Sp; F0 = false, γa = (Ch !== aCh), γp = (Ch !== pCh), γt = (Ch !== tCh))
            F0p = F0(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

            val += Γp * Π0slice[i] * F0p
        end

        return temperature(F) * val * sign
    end

    # compute K2
    SG(K2, InitFunction{2, Q}(diagram); mode)

    return nothing
end

# Note: K2 contributions to right part always vanishes
function BSE_K2!(
    K2   :: MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
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

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)
        Πslice  = view(Π, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            if is_mfRG === Val(true)
                # vertices
                F0l = F0(Ω, ν, ω, Ch, Sp) - F0(Ω, νInf, ω, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                # central part
                val += F0l * Π0slice[i] * FLr

            else
                # vertices
                Fl  = F( Ω, ν,    ω, Ch, Sp) - F(Ω, νInf, ω, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                # 1ℓ and central part
                val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
            end
        end

        return temperature(F) * val * sign
    end

    # compute K2
    SG(K2, InitFunction{2, Q}(diagram); mode)

    if Ch === tCh
        mult_add!(K2, FL.γt.K2, 2)
        mult_add!(K2, FL.γa.K2, -1)
    else
        add!(K2, get_reducible_vertex(FL, Ch).K2)
    end

    return nothing
end
