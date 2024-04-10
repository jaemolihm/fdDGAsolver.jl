# conversion between different frequency/momentum representations
#----------------------------------------------------------------------------------------------#

"""
    function convert_frequency(
        Ω  :: MatsubaraFrequency{Boson},
        ν  :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
        νp :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
           :: Type{Ch_from},
           :: Type{Ch_to}
        ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

Convert frequencies in the `Ch_from` channel representation to `Ch_to` channel representation
"""
@inline function convert_frequency(
    Ω  :: MatsubaraFrequency{Boson},
    ν  :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
    νp :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
       :: Type{Ch_from},
       :: Type{Ch_to}
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

    if Ch_from === Ch_to
        return Ω, ν, νp

    elseif Ch_from === pCh && Ch_to === tCh
        return Ω - ν - νp, νp, ν

    elseif Ch_from === pCh && Ch_to === aCh
        return ν - νp, Ω - ν, νp

    elseif Ch_from === tCh && Ch_to === pCh
        return Ω + ν + νp, νp, ν

    elseif Ch_from === tCh && Ch_to === aCh
        return νp - ν, Ω + ν, ν

    elseif Ch_from === aCh && Ch_to === pCh
        return Ω + νp + ν, Ω + νp, νp

    elseif Ch_from === aCh && Ch_to === tCh
        return ν - νp, νp, Ω + νp

    else
        throw(ArgumentError("Cannot convert from $Ch_from to $Ch_to!"))
    end
end

"""
    function convert_momentum(
        Q  :: BrillouinPoint,
        k  :: BrillouinPoint,
        kp :: BrillouinPoint,
           :: Type{Ch_from},
           :: Type{Ch_to}
        ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

Convert momenta in the `Ch_from` channel representation to `Ch_to` channel representation
"""
@inline function convert_momentum(
    Q  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch_from},
       :: Type{Ch_to}
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

    if Ch_from === Ch_to
        return Q, k, kp

    elseif Ch_from === pCh && Ch_to === tCh
        return Q - k - kp, kp, k

    elseif Ch_from === pCh && Ch_to === aCh
        return k - kp, Q - k, kp

    elseif Ch_from === tCh && Ch_to === pCh
        return Q + k + kp, kp, k

    elseif Ch_from === tCh && Ch_to === aCh
        return kp - k, Q + k, k

    elseif Ch_from === aCh && Ch_to === pCh
        return Q + kp + k, Q + kp, kp

    elseif Ch_from === aCh && Ch_to === tCh
        return k - kp, kp, Q + kp

    else
        throw(ArgumentError("Cannot convert from $Ch_from to $Ch_to!"))
    end
end