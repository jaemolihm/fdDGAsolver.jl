"""
    hubbard_bare_Green(mesh_ν, mesh_k, ::Type{Q}; μ, t, t2 = 0, t3 = 0)

Bare Green function of the square lattice Hubbard model.
* `e` : chemical potential. 0 is half filling for the nearest-neighbor hopping case.
* `t1`, `t2`, `t3` : First, second, and third nearest neighbor hopping energies.
"""
function hubbard_bare_Green(
    mesh_ν :: FMesh,
    mesh_k :: KMesh,
           :: Type{Q} = ComplexF64,
    ;
    μ, t1, t2 = 0, t3 = 0,
    ) where {Q}

    G0 = MeshFunction(mesh_ν, mesh_k; data_t = Q)

    # S.G is im * G
    for k in mesh_k
        k1, k2 = euclidean(k, mesh_k)
        εk  = -2 * t1 * (cos(k1) + cos(k2))
        εk += -4 * t2 * cos(k1) * cos(k2)
        εk += -2 * t3 * (cos(2k1) + cos(2k2))
        for ν in mesh_ν
            ν_value = plain_value(ν)
            G0[ν, k] = 1 / (im * ν_value + μ - εk)
        end
    end

    G0 :: NL_MF_G{Q}
end

export hubbard_bare_Green
