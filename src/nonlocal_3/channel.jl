# 2-particle reducible vertex in the asymptotic decomposition
#----------------------------------------------------------------------------------------------#

struct NL3_Channel{Q <: Number} <: AbstractReducibleVertex{Q}
    K1 :: NL_MF_K1{Q}
    K2 :: NL2_MF_K2{Q}
    K3 :: NL3_MF_K3{Q}

    function NL3_Channel(K1 :: NL_MF_K1{Q}, K2 :: NL2_MF_K2{Q}, K3 :: NL3_MF_K3{Q}) where {Q}
        return new{Q}(K1, K2, K3)
    end

    function NL3_Channel(
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
              :: Type{Q} = ComplexF64
        ) where {Q}

        mK1Ω = MatsubaraMesh(T, numK1, Boson)
        K1   = MeshFunction(mK1Ω, meshK; data_t = Q)
        set!(K1, 0)

        @assert numK1 >= numK2[1] "No. bosonic frequencies in K1 must be larger than no. bosonic frequencies in K2"
        @assert numK1 >= numK2[2] "No. bosonic frequencies in K1 must be larger than no. fermionic frequencies in K2"
        mK2Ω = MatsubaraMesh(T, numK2[1], Boson)
        mK2ν = MatsubaraMesh(T, numK2[2], Fermion)
        K2   = MeshFunction(mK2Ω, mK2ν, meshK, meshK; data_t = Q)
        set!(K2, 0)

        @assert all(numK2 .>= numK3) "Number of frequencies in K2 must be larger than in K3"
        mK3Ω = MatsubaraMesh(T, numK3[1], Boson)
        mK3ν = MatsubaraMesh(T, numK3[2], Fermion)
        K3   = MeshFunction(mK3Ω, mK3ν, mK3ν, meshK, meshK, meshK; data_t = Q)
        set!(K3, 0)

        return new{Q}(K1, K2, K3)
    end
end

# getter methods
function get_P_mesh(γ :: NL3_Channel) :: KMesh
    return meshes(γ.K1, Val(2))
end

function numP(γ :: NL3_Channel) :: Int64
    return length(meshes(γ.K1, Val(2)))
end

# copy
function Base.:copy(γ :: NL3_Channel{Q}) :: NL3_Channel{Q} where {Q}
    return NL3_Channel(copy(γ.K1), copy(γ.K2), copy(γ.K3))
end

# evaluator
@inline function (γ :: NL3_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    q  :: BrillouinPoint,
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)
    iP = MatsubaraFunctions.mesh_index_bc(P, get_P_mesh(γ))
    ik = MatsubaraFunctions.mesh_index_bc(k, get_P_mesh(γ))
    iq = MatsubaraFunctions.mesh_index_bc(q, get_P_mesh(γ))

    if is_inbounds(Ω, meshes(γ.K1, Val(1)))
        K1 && (val += γ.K1[Ω, iP])

        if is_inbounds(Ω, meshes(γ.K2, Val(1)))
            ν1_inbounds = is_inbounds(ν, meshes(γ.K2, Val(2)))
            ν2_inbounds = is_inbounds(ω, meshes(γ.K2, Val(2)))

            if ν1_inbounds && ν2_inbounds
                K2 && (val += γ.K2[Ω, ν, iP, ik] + γ.K2[Ω, ω, iP, iq])

                if (K3 && is_inbounds(Ω, meshes(γ.K3, Val(1))) &&
                          is_inbounds(ν, meshes(γ.K3, Val(2))) &&
                          is_inbounds(ω, meshes(γ.K3, Val(3))))
                    val += γ.K3[Ω, ν, ω, iP, ik, iq]
                end

            elseif ν1_inbounds
                K2 && (val += γ.K2[Ω, ν, iP, ik])

            elseif ν2_inbounds
                K2 && (val += γ.K2[Ω, ω, iP, iq])
            end
        end
    end

    return val
end
