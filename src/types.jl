# channels
abstract type ChannelTag end
struct pCh <: ChannelTag end
struct tCh <: ChannelTag end
struct aCh <: ChannelTag end

# spin components
abstract type SpinTag end
struct pSp <: SpinTag end  # parallel
struct xSp <: SpinTag end  # crossed
struct dSp <: SpinTag end  # density

# Mesh aliases
const FMesh = Mesh{MeshPoint{MatsubaraFrequency{Fermion}}, MatsubaraDomain}
const BMesh = Mesh{MeshPoint{MatsubaraFrequency{Boson}}, MatsubaraDomain}
const KMesh = Mesh{MeshPoint{BrillouinPoint{2}}, BrillouinDomain{2}}

# MeshFunction aliases : Local
const MF_G{Q}  = MeshFunction{1, Q, Tuple{FMesh}, Array{Q, 1}}
const MF_Π{Q}  = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
const MF_K1{Q} = MeshFunction{1, Q, Tuple{BMesh}, Array{Q, 1}}
const MF_K2{Q} = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
const MF_K3{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, FMesh}, Array{Q, 3}}

# MeshFunction aliases : Nonlocal Green function and bubble
const NL_MF_G{Q}  = MeshFunction{2, Q, Tuple{FMesh, KMesh}, Array{Q, 2}}
const NL_MF_Π{Q}  = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}

# MeshFunction aliases : Nonlocal vertex with only bosonic frequency dependence
const NL_MF_K1{Q} = MeshFunction{2, Q, Tuple{BMesh, KMesh}, Array{Q, 2}}
const NL_MF_K2{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, KMesh}, Array{Q, 3}}
const NL_MF_K3{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, FMesh, KMesh}, Array{Q, 4}}


struct InfiniteMatsubaraFrequency; end
const νInf = InfiniteMatsubaraFrequency()

Base.:+(::MatsubaraFrequency{Boson}, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::MatsubaraFrequency{Boson}, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:+(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency{Boson}) = InfiniteMatsubaraFrequency()
Base.:-(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency{Boson}) = InfiniteMatsubaraFrequency()

const kInf = BrillouinPoint(0, 0)
