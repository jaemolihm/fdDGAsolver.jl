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

# MeshFunction aliases : Nonlocal
const NL_MF_G{Q}  = MeshFunction{2, Q, Tuple{FMesh, KMesh}, Array{Q, 2}}
const NL_MF_Π{Q}  = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}



struct InfiniteMatsubaraFrequency; end
const νInf = InfiniteMatsubaraFrequency()

Base.:+(::MatsubaraFrequency{Boson}, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::MatsubaraFrequency{Boson}, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:+(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency{Boson}) = InfiniteMatsubaraFrequency()
Base.:-(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency{Boson}) = InfiniteMatsubaraFrequency()


"""
    function convert_frequency(
    Ω  :: MatsubaraFrequency{Boson},
    ν  :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
    νp :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
       :: Type{Ch_from},
       :: Type{Ch_to},
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}
Convert frequencies in the `Ch_from` channel representation to `Ch_to` channel representation.
"""
@inline function convert_frequency(
    Ω  :: MatsubaraFrequency{Boson},
    ν  :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
    νp :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
       :: Type{Ch_from},
       :: Type{Ch_to},
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

    if Ch_from === Ch_to
        return Ω, ν, νp

    elseif Ch_from === pCh && Ch_to === tCh
        return Ω - ν - νp, νp, ν

    elseif Ch_from === pCh && Ch_to === aCh
        return ν - νp, Ω - ν, νp

    elseif Ch_from === tCh && Ch_to === pCh
        return Ω + ν + νp, νp, ν

    elseif Ch_from === tCh && Ch_to === aCh
        return νp - ν, Ω + ν, ν

    elseif Ch_from === aCh && Ch_to === pCh
        return Ω + νp + ν, Ω + νp, νp

    elseif Ch_from === aCh && Ch_to === tCh
        return ν - νp, νp, Ω + νp

    else
        throw(ArgumentError("Wrong channels $Ch_from or $Ch_to"))
    end

end
