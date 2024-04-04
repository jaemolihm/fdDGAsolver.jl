
function bubbles!(
    Πpp :: MF_K2{Q},
    Πph :: MF_K2{Q},
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
    S :: ParquetSolver
    ) :: Nothing

    bubbles!(S.Πpp, S.Πph, S.G)

end
