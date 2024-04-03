# channels
abstract type ChannelTag end
struct pCh <: ChannelTag end
struct tCh <: ChannelTag end
struct aCh <: ChannelTag end

# spin components
abstract type SpinTag end
struct pSp <: SpinTag end
struct xSp <: SpinTag end

# Mesh aliases
const FMesh = Mesh{MeshPoint{MatsubaraFrequency{Fermion}}, MatsubaraDomain}
const BMesh = Mesh{MeshPoint{MatsubaraFrequency{Boson}}, MatsubaraDomain}

# MeshFunction aliases
const MF_G{Q}  = MeshFunction{1, Q, Tuple{FMesh}, Array{Q, 1}}
const MF_Π{Q}  = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
const MF_K1{Q} = MeshFunction{1, Q, Tuple{BMesh}, Array{Q, 1}}
const MF_K2{Q} = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
const MF_K3{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, FMesh}, Array{Q, 3}}


# # MatsubaraSymmetry aliases
# const MS2 = MatsubaraSymmetry{2, 0}
# const MS3 = MatsubaraSymmetry{3, 0}

# # MatsubaraSymmetryGroup aliases
# const MSG2 = MatsubaraSymmetryGroup{2, 0, 2, Float64}
# const MSG3 = MatsubaraSymmetryGroup{3, 0, 3, Float64}

# # MatsubaraInitFunction aliases
# const MIF2 = MatsubaraInitFunction{2, 0, Float64}
# const MIF3 = MatsubaraInitFunction{3, 0, Float64}
