# Note: K1 contributions to left and right part always vanish
function BSE_K1!(
    K1   :: NL_MF_K1{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
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
        Π0slice = view(Π0, Ω, :, P, :)
        Πslice  = view(Π , Ω, :, P, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i.I[1]])
            q = value(meshes(Π0, Val(4))[i.I[2]])

            if is_mfRG === Val(true)
                # mfRG for the K1 class
                # vertices
                Fl  = F0(Ω, νInf, ω, P, k0, q, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)

                # central part
                val += Fl * Π0slice[i] * FLr

            else
                # fdDΓA for the K1 class
                # vertices
                Fl  = F( Ω, νInf, ω, P, k0,  q, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)

                # 1ℓ and central part
                val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
            end
        end

        return temperature(F) * val / numP(F) * sign
    end

    # compute K1
    SG(K1, InitFunction{2, Q}(diagram); mode)

    return nothing
end



function BSE_K1_new!(
    K1   :: NL_MF_K1{Q},
    F0   :: AbstractVertex{Q},
    F    :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
    SG   :: SymmetryGroup{2, Q},
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

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(Π0, Val(2)), meshes(Π0, Val(4))), view(Π0, Ω, :, P, :))
        Πslice  = MeshFunction((meshes(Π,  Val(2)), meshes(Π,  Val(4))), view(Π,  Ω, :, P, :))

        for iq in eachindex(meshes(Πslice, Val(2))), iω in eachindex(meshes(Πslice, Val(1)))
            ω = value(meshes(Πslice, Val(1))[iω])
            q = value(meshes(Πslice, Val(2))[iq])
            Π_  = Πslice[iω, iq]
            Π0_ = Π0slice[iω, iq]

            # vertices
            Fl  = F( Ω, νInf, ω, P, k0, q, Ch, Sp)
            F0l = F0(Ω, νInf, ω, P, k0, q, Ch, Sp)

            # 1ℓ and central part
            if is_mfRG === Val(true)
                val += (Fl - F0l) * Π_ * U
            else
                val += (Fl * Π_ - F0l * Π0_) * U
            end
        end

        return temperature(F) * val / numP(F) * sign
    end

    # compute K1
    SG(K1, InitFunction{2, Q}(diagram); mode)

    return nothing
end
