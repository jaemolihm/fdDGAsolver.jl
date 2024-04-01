"""
# Frequency parametrization
The frequency parametrization reads
- a channel: ``(v1, v2; w) = ( v1  , -v1-w,  v2+w, -v2)``
- p channel: ``(v1, v2; w) = ( v1  ,  v2+w, -v1-w, -v2)``
- t channel: ``(v1, v2; w) = ( v2+w, -v1-w,  v1  , -v2)``
(Frequency is positive for an outgoing leg.)
The frequencies can be obtained by reordering the A channel frequencies using the indices
ordering shown above.
Note that compared to Gievers et al, Eur. Phys. J. B 95, 108 (2022), Fig. 3, we use opposite
sign for `w` in the P channel to make the frequencies are related to the A channel ones
via a simple exchange of 2nd and 3rd indices.
"""


@inline function shuffle_to_standard(C::Symbol, i)
    C === :a && return (i[1], i[2], i[3], i[4])
    C === :p && return (i[1], i[3], i[2], i[4])
    C === :t && return (i[3], i[2], i[1], i[4])
    throw(ArgumentError("Wrong channel $C"))
end

@inline function shuffle_to_channel(C::Symbol, i)
    C === :a && return (i[1], i[2], i[3], i[4])
    C === :p && return (i[1], i[3], i[2], i[4])
    C === :t && return (i[3], i[2], i[1], i[4])
    throw(ArgumentError("Wrong channel $C"))
end

@inline frequency_to_channel(v1, v2, w) = (v1, -v1-w, v2+w, -v2)

# v1 + v2 + v3 + v4 = 0 is assumed to hold.
@inline frequency_to_channel(v1, v2, v3, v4) = (v1, -v4, v3 + v4)

@inline frequency_to_channel(C :: Symbol, v1, v2, w) = shuffle_to_standard(C, frequency_to_channel(v1, v2, w))
@inline frequency_to_channel(C :: Symbol, v1, v2, v3, v4) = frequency_to_channel(shuffle_to_channel(C, (v1, v2, v3, v4))...)
@inline frequency_to_channel(C :: Symbol, v1234) = frequency_to_channel(C, v1234...)

@inline function convert_channel(C_from :: Symbol, C_to :: Symbol, v1, v2, w)
    if C_from === C_to
        v1, v2, w
    else
        frequency_to_channel(C_to, frequency_to_channel(C_from, v1, v2, w))
    end
end