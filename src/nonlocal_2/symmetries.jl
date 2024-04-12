# K2 symmetries without s-wave truncation
#----------------------------------------------------------------------------------------------#

function sK2_NL2_ref(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], _ref(w[3], m), _ref(w[4], m)), Operation()
end

function sK2_NL2_rot(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], _rot(w[3], m), _rot(w[4], m)), Operation()
end

# symmetries in the particle-particle channel
function sK2_NL2_pp1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], -w[2], fold_back(-w[3], m), fold_back(-w[4], m)), Operation(sgn = false, con = true)
end

function sK2_NL2_pp2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[1] - w[2], w[3], fold_back(w[3] - w[4], m)), Operation()
end

# symmetries in the particle-hole channel
function sK2_NL2_ph1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return sK2_NL2_pp1(w, m)
end

function sK2_NL2_ph2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], w[1] + w[2], fold_back(-w[3], m), fold_back(w[3] + w[4], m)), Operation()
end
