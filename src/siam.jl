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
    for ν in value.(mesh)
        if D == Inf
            G0[ν] = 1 / (value(ν) + im * e + Δ * sign(value(ν)))
        else
            G0[ν] = 1 / (value(ν) + im * e + 2Δ / π * atan(D / value(ν)))
        end
    end

    G0 :: MF_G{Q}
end