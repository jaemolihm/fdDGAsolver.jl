
function class_inbounds(
    class      :: Vector{Tuple{Int, Operation}},
    symmetries :: Vector{Symmetry{DD}},
    f          :: MeshFunction{DD},
    )          :: Bool where {DD}
    # False if any symmetry in symmetries map any element in class outside the bounds of f.

    for (idx, op) in class
        w = value.(to_meshes(f, idx))

        for S in symmetries
            wp, _ = S(w)
            if ! MatsubaraFunctions._all_inbounds(f, wp...)
                return false
            end
        end
    end

    return true
end

function my_SymmetryGroup(
    symmetries :: Vector{Symmetry{DD}},
    f          :: MeshFunction{DD}
    ) where {DD}
    # SymmetryGroup with elements that strictly preserves the symmetry.
    # Remove a class if any element maps outside the box by any symmetry.

    SG = SymmetryGroup(symmetries, f)
    filter!(class -> class_inbounds(class, symmetries, f), SG.classes)
    SG
end

function my_symmetrize!(
    f  :: MeshFunction{DD, Q},
    SG :: SymmetryGroup{DD, Q}
    ) where {DD, Q <: Number}

    # Symmetrize MeshFunction f, set elements not in SG.classes to zero.

    err = zero(real(Q))

    # Store the representative value of f for each class
    ref_list = [f[class[1][1]] for class in SG.classes]
    set!(f, 0)

    for (class, ref) in zip(SG.classes, ref_list)
        for (idx, op) in class
            new_val = op(ref)
            new_err = abs(f[idx] - new_val)

            err = max(err, new_err)
            f[idx] = new_val
        end
    end

    return err
end
