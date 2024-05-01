function BSE_L_K3!(
    K3       :: MF_K3{Q},
    cache_Γ  :: MF_K3{Q},
    cache_F0 :: MF_K3{Q},
    Π0       :: MF_Π{Q},
    Π        :: MF_Π{Q},
    SG       :: SymmetryGroup{3, Q},
    sign     :: Int,
             :: Type{Ch},
             :: Type{Sp},
    is_mfRG  :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode     :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    T = temperature(meshes(K3, Val(1)))

    # model the diagram
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val     = zero(Q)
        Γslice  = view(cache_Γ,  Ω, ν,  :)
        F0slice = view(cache_F0, Ω, :, νp)

        for i in eachindex(Γslice)
            ω = value(meshes(cache_Γ, Val(3))[i])
            Π0_val = Π0[Ω, ω]

            val += Γslice[i] * Π0_val * F0slice[i]
        end

        return T * val * sign
    end

    # compute K3
    SG(K3, InitFunction{3, Q}(diagram); mode)

    return nothing
end


function BSE_K3!(
    K3       :: MF_K3{Q},
    FL       :: AbstractVertex{Q},
    cache_Γ  :: MF_K3{Q},
    cache_F  :: MF_K3{Q},
    cache_F0 :: MF_K3{Q},
    Π0       :: MF_Π{Q},
    Π        :: MF_Π{Q},
    SG       :: SymmetryGroup{3, Q},
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

        Ω, ν, νp = wtpl
        val = zero(Q)

        if Ch === aCh || Ch === tCh
            Γslice  = view(cache_Γ,  Ω, νp,  :)  # using symmetry Γ[Ω, ω, νp] = Γ[Ω, νp, ω]
        elseif Ch === pCh
            Γslice  = view(cache_Γ,  Ω, :, νp)
        end

        γL = get_reducible_vertex(FL, Ch)

        Fslice  = view(cache_F,  Ω,  ν,  :)
        F0slice = view(cache_F0, Ω,  :, νp)

        for i in eachindex(Fslice)
            ω = value(meshes(cache_F, Val(3))[i])
            Π0_val = Π0[Ω, ω]
            Π_val  = Π[ Ω, ω]

            if is_mfRG === Val(true)
                # mfRG for the K3 class

                # right part
                val += Fslice[i] * Π0_val * Γslice[i] * sign1

                # central part
                ω_ = _crossing(Ω, ω, Ch)
                if is_inbounds(ω_, meshes(S.FL.γt.K3, Val(2)))
                    if Ch === aCh || Ch === pCh
                        val += Fslice[i] * Π0_val * γL.K3[Ω, ω_, νp] * sign2
                    elseif Ch === tCh
                        val += Fslice[i] * Π0_val * (2 * FL.γt.K3[Ω, ω_, νp] - FL.γa.K3[Ω, ω_, νp]) * sign2
                    end
                end

            else
                # fdDΓA for the K3 class
                # 1ℓ and right part
                val += Fslice[i] * ((Π_val - Π0_val) * F0slice[i] + Π_val * Γslice[i]) * sign1

                # central part
                ω_ = _crossing(Ω, ω, Ch)
                if is_inbounds(ω_, meshes(γL.K3, Val(2)))
                    if Ch === aCh || Ch === pCh
                        val += Fslice[i] * Π_val * γL.K3[Ω, ω_, νp] * sign2
                    elseif Ch === tCh
                        val += Fslice[i] * Π_val * (2 * FL.γt.K3[Ω, ω_, νp] - FL.γa.K3[Ω, ω_, νp]) * sign2
                    end
                end
            end
        end

        if Ch === aCh || Ch === pCh
            return temperature(FL) * val + γL.K3[Ω, ν, νp]
        else
            return temperature(FL) * val + 2 * FL.γt.K3[Ω, ν, νp] - FL.γa.K3[Ω, ν, νp]
        end

    end

    # compute K3
    SG(K3, InitFunction{3, Q}(diagram); mode)

    return nothing
end
