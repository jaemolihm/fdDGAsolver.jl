
"""
    function convert_frequency(
    Ω  :: MatsubaraFrequency{Boson},
    ν  :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
    νp :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
       :: Type{Ch_from},
       :: Type{Ch_to},
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}
Convert frequencies in the `Ch_from` channel representation to `Ch_to` channel representation.
"""
@inline function convert_frequency(
    Ω  :: MatsubaraFrequency{Boson},
    ν  :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
    νp :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
       :: Type{Ch_from},
       :: Type{Ch_to},
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
        throw(ArgumentError("Wrong channels $Ch_from or $Ch_to"))
    end

end

@inline function convert_momentum(
    Ω  :: BrillouinPoint,
    ν  :: BrillouinPoint,
    νp :: BrillouinPoint,
       :: Type{Ch_from},
       :: Type{Ch_to},
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

    # Momentum convention is the same as the frequency convention

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
        throw(ArgumentError("Wrong channels $Ch_from or $Ch_to"))
    end

end
