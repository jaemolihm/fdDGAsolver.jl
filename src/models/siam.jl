"""
    siam_bare_Green(mesh, ::Type{Q}; Δ, e, D)

Bare Green function of the single impurity Anderson model.
* `Δ` : Hybridization
* `e` : onsite energy. 0 is half filling.
* `D` : Bandwidth
"""
function siam_bare_Green(
    mesh :: FMesh,
         :: Type{Q} = ComplexF64,
    ;
    Δ, e, D,
    ) where {Q}

    G0 = MeshFunction(mesh; data_t = Q)

    # S.G is im * G
    for ν in mesh
        ν_value = plain_value(ν)
        if D == Inf
            G0[ν] = 1 / (ν_value + im * e + Δ * sign(ν_value))
        else
            G0[ν] = 1 / (ν_value + im * e + 2Δ / π * atan(D / ν_value))
        end
    end

    G0 :: MF_G{Q}
end

export siam_bare_Green
