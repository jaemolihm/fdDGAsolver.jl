
function bubbles!(
    Πpp :: MF_Π{Q},
    Πph :: MF_Π{Q},
    G   :: MF_G{Q},
    ) :: Nothing where {Q}

    for iΩ in eachindex(meshes(Πpp, 1)), iν in eachindex(meshes(Πpp, 2))
        Ω = value(meshes(Πpp, 1)[iΩ])
        ν = value(meshes(Πpp, 2)[iν])
        Πpp[iΩ, iν] = G(ν) * G(Ω - ν)
        Πph[iΩ, iν] = G(Ω + ν) * G(ν)
    end

    return nothing
end

function bubbles!(
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    G   :: NL_MF_G{Q},
    ) :: Nothing where {Q}

    for iP in eachindex(meshes(Πpp, 3)), ik in eachindex(meshes(Πpp, 4))
        P = value(meshes(Πpp, 3)[iP])
        k = value(meshes(Πpp, 4)[ik])

        for iΩ in eachindex(meshes(Πpp, 1)), iν in eachindex(meshes(Πpp, 2))
            Ω = value(meshes(Πpp, 1)[iΩ])
            ν = value(meshes(Πpp, 2)[iν])
            Πpp[iΩ, iν, iP, ik] = G(ν, k) * G(Ω - ν, P - k)
            Πph[iΩ, iν, iP, ik] = G(Ω + ν, P + k) * G(ν, k)
        end
    end

    return nothing
end



function bubbles!(
    S :: ParquetSolver
    ) :: Nothing

    bubbles!(S.Πpp, S.Πph, S.G)

end
