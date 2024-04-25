# diagrammatic channels
#----------------------------------------------------------------------------------------------#

"""
    abstract type ChannelTag end

Abstract type for diagrammatic channels
"""
abstract type ChannelTag end

"""
    struct pCh <: ChannelTag end

Parallel channel
"""
struct pCh <: ChannelTag end

"""
    struct tCh <: ChannelTag end

Transversal channel
"""
struct tCh <: ChannelTag end

"""
    struct aCh <: ChannelTag end

Antiparallel channel
"""
struct aCh <: ChannelTag end

# spin components
#----------------------------------------------------------------------------------------------#
# We define two spin components: parallel `pSp` and crossed `xSp`.
# All vertices are stored in the parallel spin component, and the crossed component is computed
# on the fly using the following crossing symmetry relations.
# ``Γp{xSp}(Ω, ν, ω) = -Γp{pSp}(Ω, ν, Ω - ω) = -Γp{pSp}(Ω, Ω - ν, ω)``
# ``Γt{xSp}(Ω, ν, ω) = -Γa{pSp}(Ω, ω, ν)``
# ``Γa{xSp}(Ω, ν, ω) = -Γt{pSp}(Ω, ω, ν)``

# We also define the density `dSp` component, and use it to simplify some evaluations.
# The density component is computed on the fly using the relation ``dSp = 2 * pSp + xSp``.
# Since the t-channel BSE is diagonal in the `xSp` and `dSp` channels but not in the
# `pSp` channel, we compute the t-channel BSE in `dSp`, and then subtract the `xSp` contribution
# (which is -1 times the `pSp` term in a channel by crossing symmetry) and divide by 2 to
# get the `pSp` component.

# The physical spin basis in terms of singlet (S), triplet (T), density (D), and magnetic (M)
# channels is given by
# ``S = pSp - xSp``
# ``T = pSp + xSp``
# ``D = 2 * pSp + xSp = dSp``
# ``M = xSp``

# pCh, pSp =  1/2 * (pCh, S) + 1/2 * (pCh, T)
# aCh, pSp = -1   * (tCh, M)
# tCh, pSp =  1/2 * (tCh, D) - 1/2 * (tCh, M)

# pCh, xSp = -1/2 * (pCh, S) + 1/2 * (pCh, T)
# aCh, xSp = -1/2 * (tCh, D) + 1/2 * (tCh, M)
# tCh, xSp = (tCh, M)

# pCh, dSp =  1/2 * (pCh, S) + 3/2 * (pCh, T)
# aCh, dSp = -1/2 * (tCh, D) - 3/2 * (tCh, M)
# tCh, dSp = (tCh, D)

"""
    abstract type SpinTag end

Abstract type for spin components
"""
abstract type SpinTag end

"""
    struct pSp <: SpinTag end

Parallel spin component
"""
struct pSp <: SpinTag end

"""
    struct xSp <: SpinTag end

Crossed spin component
"""
struct xSp <: SpinTag end

"""
    struct dSp <: SpinTag end

Density component
"""
struct dSp <: SpinTag end

# Mesh aliases
#----------------------------------------------------------------------------------------------#

const FMesh = Mesh{MeshPoint{MatsubaraFrequency{Fermion}}, MatsubaraDomain}
const BMesh = Mesh{MeshPoint{MatsubaraFrequency{Boson}}, MatsubaraDomain}
const KMesh = Mesh{MeshPoint{BrillouinPoint{2}}, BrillouinDomain{2, 4}}

# MeshFunction aliases : local
#----------------------------------------------------------------------------------------------#

const MF_G{Q}  = MeshFunction{1, Q, Tuple{FMesh}, Array{Q, 1}}
const MF_Π{Q}  = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
const MF_K1{Q} = MeshFunction{1, Q, Tuple{BMesh}, Array{Q, 1}}
const MF_K2{Q} = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
const MF_K3{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, FMesh}, Array{Q, 3}}

# MeshFunction aliases : nonlocal Green's function and bubble
#----------------------------------------------------------------------------------------------#

const NL_MF_G{Q} = MeshFunction{2, Q, Tuple{FMesh, KMesh}, Array{Q, 2}}
const NL_MF_Π{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, KMesh}, Array{Q, 3}}

# MeshFunction aliases : nonlocal vertex with only bosonic frequency dependence
#----------------------------------------------------------------------------------------------#

const NL_MF_K1{Q} = MeshFunction{2, Q, Tuple{BMesh, KMesh}, Array{Q, 2}}
const NL_MF_K2{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, KMesh}, Array{Q, 3}}
const NL_MF_K3{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, FMesh, KMesh}, Array{Q, 4}}

# MeshFunction aliases : nonlocal vertex with bosonic and fermionic frequency dependences
#----------------------------------------------------------------------------------------------#

const NL2_MF_Π{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}
const NL2_MF_K2{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}
const NL2_MF_K3{Q} = MeshFunction{5, Q, Tuple{BMesh, FMesh, FMesh, KMesh, KMesh}, Array{Q, 5}}

# struct to describe high-frequency limit
#----------------------------------------------------------------------------------------------#

struct InfiniteMatsubaraFrequency end
const νInf = InfiniteMatsubaraFrequency()

Base.:+(::MatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::MatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:+(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:+(::InfiniteMatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::InfiniteMatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()

MatsubaraFunctions.is_inbounds(::InfiniteMatsubaraFrequency, ::Mesh) = false

# origin of the Brillouin zone
#----------------------------------------------------------------------------------------------#

const k0 = BrillouinPoint(0, 0)
