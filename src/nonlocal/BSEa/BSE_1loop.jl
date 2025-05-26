# Note: K1 contributions to left and right part always vanish
function BSE_K1_1loop!(
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

                # 1ℓ and central part
                val += Fl * (Πslice[i] - Π0slice[i]) * F0r
            end
        end

        return temperature(F) * val * sign
    end

    # compute K1
    SG(K1, InitFunction{2, Q}(diagram); mode)

    return nothing
end


function BSE_K2_1loop!(
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

                # 1ℓ and central part
                val += Fl * (Πslice[iω] - Π0slice[iω]) * F0r

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


function BSE_K3_1loop!(
    K3       :: NL_MF_K3{Q},
    FL       :: AbstractVertex{Q},
    cache_Γ  :: NL_MF_K3{Q},
    cache_F  :: NL_MF_K3{Q},
    cache_F0 :: NL_MF_K3{Q},
    Π0       :: NL_MF_Π{Q},
    Π        :: NL_MF_Π{Q},
    SG       :: SymmetryGroup{4, Q},
    sign1    :: Int,
    sign2    :: Int,
             :: Type{Ch},
             :: Type{Sp},
    is_mfRG  :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode     :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val = zero(Q)

        if Ch === aCh || Ch === tCh
            Γslice  = view(cache_Γ,  Ω, νp,  :, P)  # using symmetry Γ[Ω, ω, νp] = Γ[Ω, νp, ω]
        elseif Ch === pCh
            Γslice  = view(cache_Γ,  Ω, :, νp, P)
        end

        γL = get_reducible_vertex(FL, Ch)

        Fslice  = view(cache_F,  Ω,  ν,  :, P)
        F0slice = view(cache_F0, Ω,  :, νp, P)

        for i in eachindex(Fslice)
            ω = value(meshes(cache_F, Val(3))[i])
            Π0_val = Π0[Ω, ω, P]
            Π_val  = Π[ Ω, ω, P]

            if is_mfRG === Val(true)
                # mfRG for the K3 class

                # right part
                val += Fslice[i] * Π0_val * Γslice[i] * sign1

                # central part
                ω_ = _crossing(Ω, ω, Ch)
                if is_inbounds(ω_, meshes(γL.K3, Val(2)))
                    if Ch === aCh || Ch === pCh
                        val += Fslice[i] * Π0_val * γL.K3[Ω, ω_, νp, P] * sign2
                    elseif Ch === tCh
                        val += Fslice[i] * Π0_val * (2 * FL.γt.K3[Ω, ω_, νp, P] - FL.γa.K3[Ω, ω_, νp, P]) * sign2
                    end
                end

            else
                # fdDΓA for the K3 class
                # 1ℓ term
                val += Fslice[i] * (Π_val - Π0_val) * F0slice[i] * sign1
            end
        end

        if Ch === aCh || Ch === pCh
            return temperature(FL) * val# + γL.K3[Ω, ν, νp, P]
        else
            return temperature(FL) * val# + 2 * FL.γt.K3[Ω, ν, νp, P] - FL.γa.K3[Ω, ν, νp, P]
        end

    end

    # compute K3
    SG(K3, InitFunction{4, Q}(diagram); mode)

    return nothing
end
