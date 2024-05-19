struct NL3_Vertex{Q, VT} <: AbstractNonlocalVertex{Q}
    F0 :: VT
    γp :: NL3_Channel{Q}
    γt :: NL3_Channel{Q}
    γa :: NL3_Channel{Q}

    function NL3_Vertex(
        F0 :: VT,
        γp :: NL3_Channel{Q},
        γt :: NL3_Channel{Q},
        γa :: NL3_Channel{Q},
        )  :: NL3_Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    function NL3_Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
        ) where {VT}

        Q = eltype(F0)

        γ = NL3_Channel(T, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: NL3_Vertex{Q}
    end
end

channel_type(::Type{NL3_Vertex}) = NL3_Channel


function Base.show(io::IO, Γ::NL3_Vertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(bare_vertex(Γ.F0)), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

# copy
function Base.:copy(
    F :: NL3_Vertex{Q}
    ) :: NL3_Vertex{Q} where {Q}

    return NL3_Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

function average_fermionic_momenta!(K2 :: NL2_MF_K2)
    K2_avg = sum(K2.data, dims=4) ./ size(K2.data, 4)
    @views for i in axes(K2.data, 4)
        K2.data[:, :, :, i] .= K2_avg
    end
    return nothing
end


function average_fermionic_momenta!(K3 :: NL3_MF_K3)
    K3_avg = sum(K3.data, dims=(5, 6)) ./ size(K3.data, 5) ./ size(K3.data, 6)
    @views for j in axes(K3.data, 6), i in axes(K3.data, 5)
        K3.data[:, :, :, :, i, j] .= K3_avg
    end
    return nothing
end
