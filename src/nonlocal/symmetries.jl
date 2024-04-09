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
    )
    # Complex conjugation
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

function sK1_conj(
    w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}},
    )
    # Complex conjugation
    return (-w[1], -w[2]), Operation(sgn = false, con = true)
end

function sK1_ref(
    w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], _ref(w[2], m)), Operation()
end

function sK1_rot(
    w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], _rot(w[2], m)), Operation()
end

function sK2_ref(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[2], _ref(w[3], m)), Operation()
end

function sK2_rot(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[2], _rot(w[3], m)), Operation()
end

function sK3_ref(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[2], w[3], _ref(w[4], m)), Operation()
end

function sK3_rot(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[2], w[3], _rot(w[4], m)), Operation()
end

# Symmetries in the particle-particle channel
function sK2pp1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (-w[1], -w[2], fold_back(-w[3], m)), Operation(sgn = false, con = true)
end

function sK2pp2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[1] - w[2], w[3]), Operation()
end

function sK3pp1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (-w[1], -w[2], -w[3], fold_back(-w[4], m)), Operation(sgn = false, con = true)
end

function sK3pp2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[3], w[2], w[4]), Operation()
end

function sK3pp3(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[1] - w[2], w[1] - w[3], w[4]), Operation()
end

# particle-hole symmetries

function sK2ph1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return sK2pp1(w, m)
end

function sK2ph2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (-w[1], w[1] + w[2], fold_back(-w[3], m)), Operation()
end

function sK3ph1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return sK3pp1(w, m)
end

function sK3ph2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return sK3pp2(w, m)
end

function sK3ph3(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (-w[1], w[1] + w[2], w[1] + w[3], fold_back(-w[4], m)), Operation()
end



# Bubbles
function sΠ_ref(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[2], _ref(w[3], m), _ref(w[4], m)), Operation()
end

function sΠ_rot(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[2], _rot(w[3], m), _rot(w[4], m)), Operation()
end


function sΠ_pp1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (-w[1], -w[2], fold_back(-w[3], m), fold_back(-w[4], m)), Operation(sgn = false, con = true)
end

function sΠ_pp2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (w[1], w[1] - w[2], w[3], fold_back(w[3] - w[4], m)), Operation()
end


function sΠ_ph1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh,
    )
    return sΠ_pp1(w, m)
end

function sΠ_ph2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh,
    )
    return (-w[1], w[1] + w[2], fold_back(-w[3], m), fold_back(w[3] + w[4], m)), Operation()
end
