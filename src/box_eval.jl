# This generic implementation is slower than explicitly listing DD values. Not sure why...

# @inline function box_eval(
#     f :: MeshFunction{DD, Q},
#     w :: Vararg{MatsubaraFrequency, DD}
#     ) :: Q where {DD, Q}

#     if any(ntuple(i -> !is_inbounds(w[i], meshes(f, i)), DD))
#         return zero(Q)
#     else
#         return f[w...]
#     end
# end


@inline function box_eval(
    f  :: MeshFunction{1, Q},
    w1 :: AbstractValue,
    ) :: Q where {Q}

    is_inbounds(w1, meshes(f, 1)) || return zero(Q)
    return f[w1]
end

@inline function box_eval(
    f  :: MeshFunction{2, Q},
    w1 :: AbstractValue,
    w2 :: AbstractValue,
    ) :: Q where {Q}

    is_inbounds(w1, meshes(f, 1)) || return zero(Q)
    is_inbounds(w2, meshes(f, 2)) || return zero(Q)
    return f[w1, w2]
end


@inline function box_eval(
    f  :: MeshFunction{3, Q},
    w1 :: AbstractValue,
    w2 :: AbstractValue,
    w3 :: AbstractValue,
    ) :: Q where {Q}

    is_inbounds(w1, meshes(f, 1)) || return zero(Q)
    is_inbounds(w2, meshes(f, 2)) || return zero(Q)
    is_inbounds(w3, meshes(f, 3)) || return zero(Q)
    return f[w1, w2, w3]
end

@inline function box_eval(
    f  :: MeshFunction{4, Q},
    w1 :: AbstractValue,
    w2 :: AbstractValue,
    w3 :: AbstractValue,
    w4 :: AbstractValue,
    ) :: Q where {Q}

    is_inbounds(w1, meshes(f, 1)) || return zero(Q)
    is_inbounds(w2, meshes(f, 2)) || return zero(Q)
    is_inbounds(w3, meshes(f, 3)) || return zero(Q)
    is_inbounds(w4, meshes(f, 4)) || return zero(Q)
    return f[w1, w2, w3, w4]
end


@inline function box_eval(
    f  :: MeshFunction{5, Q},
    w1 :: AbstractValue,
    w2 :: AbstractValue,
    w3 :: AbstractValue,
    w4 :: AbstractValue,
    w5 :: AbstractValue,
    ) :: Q where {Q}

    is_inbounds(w1, meshes(f, 1)) || return zero(Q)
    is_inbounds(w2, meshes(f, 2)) || return zero(Q)
    is_inbounds(w3, meshes(f, 3)) || return zero(Q)
    is_inbounds(w4, meshes(f, 4)) || return zero(Q)
    is_inbounds(w5, meshes(f, 5)) || return zero(Q)
    return f[w1, w2, w3, w4, w5]
end
