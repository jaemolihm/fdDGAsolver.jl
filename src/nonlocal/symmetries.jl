# Spatial symmetries
function _ref(k :: BrillouinPoint{2}, m :: KMesh) :: BrillouinPoint{2}
    # Reflection along zone diagonal (x, y) -> (y, x)
    return BrillouinPoint(k[2], k[1])
end

function _rot(k :: BrillouinPoint{2}, m :: KMesh) :: BrillouinPoint{2}
    # rotation by π/2 (x, y) -> (y, -x)
    return fold_back(BrillouinPoint(k[2], -k[1]), m)
end

# self-energy symmetries
function sΣ_conj(
    w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    # We store iΣ, so the symmetry Σ -> Σ* becomes iΣ -> -(iΣ)*.
    return (-w[1], -w[2]), Operation(sgn = true, con = true)
end

function sΣ_ref(
    w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], _ref(w[2], m)), Operation()
end

function sΣ_rot(
    w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], _rot(w[2], m)), Operation()
end

# # particle-particle symmetries
# function sK1pp(
#     w :: NTuple{1, MatsubaraFrequency},
#     ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation}

#     return (-w[1],), Operation(sgn = false, con = true)
# end

# function sK2pp1(
#     w :: NTuple{2, MatsubaraFrequency},
#     ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation}

#     return (-w[1], -w[2]), Operation(sgn = false, con = true)
# end

# function sK2pp2(
#     w :: NTuple{2, MatsubaraFrequency},
#     ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation}

#     return (w[1], w[1] - w[2]), Operation()
# end

# function sK3pp1(
#     w :: NTuple{3, MatsubaraFrequency},
#     ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

#     return (-w[1], -w[2], -w[3]), Operation(sgn = false, con = true)
# end

# function sK3pp2(
#     w :: NTuple{3, MatsubaraFrequency},
#     ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

#     return (w[1], w[3], w[2]), Operation()
# end

# function sK3pp3(
#     w :: NTuple{3, MatsubaraFrequency},
#     ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

#     return (w[1], w[1] - w[2], w[1] - w[3]), Operation()
# end

# # particle-hole symmetries
# function sK1ph(
#     w :: NTuple{1, MatsubaraFrequency},
#     ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation}

#     return sK1pp(w)
# end

# function sK2ph1(
#     w :: NTuple{2, MatsubaraFrequency},
#     ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation}

#     return sK2pp1(w)
# end

# function sK2ph2(
#     w :: NTuple{2, MatsubaraFrequency},
#     ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation}

#     return (-w[1], w[1] + w[2]), Operation()
# end

# function sK3ph1(
#     w :: NTuple{3, MatsubaraFrequency},
#     ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

#     return sK3pp1(w)
# end

# function sK3ph2(
#     w :: NTuple{3, MatsubaraFrequency},
#     ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

#     return sK3pp2(w)
# end

# function sK3ph3(
#     w :: NTuple{3, MatsubaraFrequency},
#     ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

#     return (-w[1], w[1] + w[2], w[1] + w[3]), Operation()
# end
