# K3 symmetries without s-wave truncation
#----------------------------------------------------------------------------------------------#

# Spatial symmetries
function sK3_NL3_ref(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], w[3], _ref(w[4], m), _ref(w[5], m), _ref(w[6], m)), Operation()
end

function sK3_NL3_rot(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], w[3], _rot(w[4], m), _rot(w[5], m), _rot(w[6], m)), Operation()
end


# symmetries in the particle-particle channel
function sK3_NL3_pp1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], -w[2], -w[3], fold_back(-w[4], m), fold_back(-w[5], m), fold_back(-w[6], m)), Operation(sgn = false, con = true)
end

function sK3_NL3_pp2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[3], w[2], w[4], w[6], w[5]), Operation()
end

function sK3_NL3_pp3(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[1] - w[2], w[1] - w[3], w[4], fold_back(w[4] - w[5], m), fold_back(w[4] - w[6], m)), Operation()
end

# symmetries in the particle-hole channel
function sK3_NL3_ph1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return sK3_NL3_pp1(w, m)
end

function sK3_NL3_ph2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return sK3_NL3_pp2(w, m)
end

function sK3_NL3_ph3(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], w[1] + w[2], w[1] + w[3], fold_back(-w[4], m), fold_back(w[4] + w[5], m), fold_back(w[4]+ w[6], m)), Operation()
end
