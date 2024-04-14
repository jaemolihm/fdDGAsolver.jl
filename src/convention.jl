# conversion between different frequency/momentum representations
#----------------------------------------------------------------------------------------------#

@inline function _convert_channel(
    A,
    b,
    bp,
    :: Type{Ch_from},
    :: Type{Ch_to}
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

    if Ch_from === Ch_to
        return A, b, bp

    elseif Ch_from === pCh && Ch_to === tCh
        return A - b - bp, bp, b

    elseif Ch_from === pCh && Ch_to === aCh
        return b - bp, A - b, bp

    elseif Ch_from === tCh && Ch_to === pCh
        return A + b + bp, bp, b

    elseif Ch_from === tCh && Ch_to === aCh
        return bp - b, A + b, b

    elseif Ch_from === aCh && Ch_to === pCh
        return A + bp + b, A + bp, bp

    elseif Ch_from === aCh && Ch_to === tCh
        return b - bp, bp, A + bp

    else
        throw(ArgumentError("Cannot convert from $Ch_from to $Ch_to !"))
    end

end

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

    _convert_channel(Ω, ν, νp, Ch_from, Ch_to)
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

    _convert_channel(Q, k, kp, Ch_from, Ch_to)
end
