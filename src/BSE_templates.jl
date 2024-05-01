# Generic implementation of the BSE in differernt channels


_crossing(Ω, ω, :: Type{pCh}) = Ω - ω
_crossing(Ω, ω, :: Type{aCh}) = ω
_crossing(Ω, ω, :: Type{tCh}) = ω


#----------------------------------------------------------------------------------------------#
# K1 class, full BSE

function BSE_K1!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1!(S.Fbuff.γa.K1, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[1], +1, aCh, pSp, is_mfRG; S.mode)
end

function BSE_K1!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1!(S.Fbuff.γp.K1, S.F0, S.F, S.FL, S.Π0pp, S.Πpp, S.SGpp[1], +1, pCh, pSp, is_mfRG; S.mode)
end

function BSE_K1!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1!(S.Fbuff.γt.K1, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[1], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K1 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K1, S.Fbuff.γa.K1)
    S.Fbuff.γt.K1.data ./= 2

    return nothing
end


#----------------------------------------------------------------------------------------------#
# K2 class, left part of the BSE

function BSE_L_K2!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K2!(S.FL.γa.K2, S.F0, S.F, S.Π0ph, S.Πph, S.SGphL[2], +1, aCh, pSp, is_mfRG; S.mode)
end

function BSE_L_K2!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K2!(S.FL.γp.K2, S.F0, S.F, S.Π0pp, S.Πpp, S.SGppL[2], +1, pCh, pSp, is_mfRG; S.mode)
end

function BSE_L_K2!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K2!(S.FL.γt.K2, S.F0, S.F, S.Π0ph, S.Πph, S.SGphL[2], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.FL.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.FL.γt.K2, S.FL.γa.K2)
    S.FL.γt.K2.data ./= 2

    return nothing
end


#----------------------------------------------------------------------------------------------#
# K2 class, full BSE

function BSE_K2!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K2!(S.Fbuff.γa.K2, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[2], +1, aCh, pSp, is_mfRG; S.mode)
end

function BSE_K2!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K2!(S.Fbuff.γp.K2, S.F0, S.F, S.FL, S.Π0pp, S.Πpp, S.SGpp[2], +1, pCh, pSp, is_mfRG; S.mode)
end

function BSE_K2!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K2!(S.Fbuff.γt.K2, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[2], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K2 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K2, S.Fbuff.γa.K2)
    S.Fbuff.γt.K2.data ./= 2

    return nothing
end


#----------------------------------------------------------------------------------------------#
# K3 class, left part of the BSE

function BSE_L_K3!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K3!(S.FL.γa.K3, S.cache_Γa, S.cache_F0a, S.Π0ph, S.Πph, S.SGphL[3], +1, aCh, pSp, is_mfRG; S.mode)
end

function BSE_L_K3!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K3!(S.FL.γp.K3, S.cache_Γpp, S.cache_F0p, S.Π0pp, S.Πpp, S.SGppL[3], -1, pCh, pSp, is_mfRG; S.mode)
end

function BSE_L_K3!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K3!(S.FL.γt.K3, S.cache_Γt, S.cache_F0t, S.Π0ph, S.Πph, S.SGphL[3], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.FL.γt.K3 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.FL.γt.K3, S.FL.γa.K3)
    S.FL.γt.K3.data ./= 2

    return nothing
end

#----------------------------------------------------------------------------------------------#
# K3 class, full BSE

function BSE_K3!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K3!(S.Fbuff.γa.K3, S.FL, S.cache_Γa, S.cache_Fa, S.cache_F0a, S.Π0ph, S.Πph, S.SGph[3], +1, +1, aCh, pSp, is_mfRG; S.mode)
end

function BSE_K3!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K3!(S.Fbuff.γp.K3, S.FL, S.cache_Γpx, S.cache_Fp, S.cache_F0p, S.Π0pp, S.Πpp, S.SGpp[3], -1, +1, pCh, pSp, is_mfRG; S.mode)
end

function BSE_K3!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K3!(S.Fbuff.γt.K3, S.FL, S.cache_Γt, S.cache_Ft, S.cache_F0t, S.Π0ph, S.Πph, S.SGph[3], -1, -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K3 has γtd = 2 γtp + γtx = 2 γtp - γax
    # We want to store γtp = (γtd + γax) / 2
    add!(S.Fbuff.γt.K3, S.Fbuff.γa.K3)
    S.Fbuff.γt.K3.data ./= 2

    return nothing
end
