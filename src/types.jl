# channel dispatch
abstract type Channel end 
struct ch_S <: Channel end 
struct ch_T <: Channel end
struct ch_D <: Channel end
struct ch_M <: Channel end

# MatsubaraFunction aliases
const MF1 = MatsubaraFunction{1, 0, 1, Float64}
const MF2 = MatsubaraFunction{2, 0, 2, Float64}
const MF3 = MatsubaraFunction{3, 0, 3, Float64}

# MatsubaraSymmetry aliases
const MS2 = MatsubaraSymmetry{2, 0}
const MS3 = MatsubaraSymmetry{3, 0}

# MatsubaraSymmetryGroup aliases
const MSG2 = MatsubaraSymmetryGroup{2, 0, 2, Float64}
const MSG3 = MatsubaraSymmetryGroup{3, 0, 3, Float64}

# MatsubaraInitFunction aliases
const MIF2 = MatsubaraInitFunction{2, 0, Float64}
const MIF3 = MatsubaraInitFunction{3, 0, Float64}