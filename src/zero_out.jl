function zero_out_points_outside_box!(
    f :: MeshFunction{DD},
    symmetries :: Vector{Symmetry{DD}},
    ) where {DD}

    # Zero out points that map outside the box by symmetry

    Threads.@threads for idx in eachindex(f.data)
        w = value.(to_meshes(f, idx))

        for symmetry in symmetries
            wp, = symmetry(w)
            if ! MatsubaraFunctions._all_inbounds(f, wp...)
                f[idx] = 0
                break
            end
        end
    end
end


function zero_out_points_outside_box!(
    S :: NL2_ParquetSolver
)
    mK_Γ = get_P_mesh(S.F)
    symmetries_K1p = [Symmetry{2}(w -> sK1_conj(w, mK_Γ)),]
    symmetries_K2p = [Symmetry{4}(w -> sK2_NL2_pp1(w, mK_Γ)),
                      Symmetry{4}(w -> sK2_NL2_pp2(w, mK_Γ)),]
    symmetries_K3p = [Symmetry{4}(w -> sK3pp1(w, mK_Γ)),
                      Symmetry{4}(w -> sK3pp2(w, mK_Γ)),
                      Symmetry{4}(w -> sK3pp3(w, mK_Γ)),]
    symmetries_K1a = [Symmetry{2}(w -> sK1_conj(w, mK_Γ)),]
    symmetries_K2a = [Symmetry{4}(w -> sK2_NL2_ph1(w, mK_Γ)),
                      Symmetry{4}(w -> sK2_NL2_ph2(w, mK_Γ)),]
    symmetries_K3a = [Symmetry{4}(w -> sK3ph1(w, mK_Γ)),
                      Symmetry{4}(w -> sK3ph2(w, mK_Γ)),
                      Symmetry{4}(w -> sK3ph3(w, mK_Γ)),]

    zero_out_points_outside_box!(S.F.γp.K1, symmetries_K1p)
    zero_out_points_outside_box!(S.F.γp.K2, symmetries_K2p)
    zero_out_points_outside_box!(S.F.γp.K3, symmetries_K3p)
    zero_out_points_outside_box!(S.F.γa.K1, symmetries_K1a)
    zero_out_points_outside_box!(S.F.γa.K2, symmetries_K2a)
    zero_out_points_outside_box!(S.F.γa.K3, symmetries_K3a)
    zero_out_points_outside_box!(S.F.γt.K1, symmetries_K1a)
    zero_out_points_outside_box!(S.F.γt.K2, symmetries_K2a)
    zero_out_points_outside_box!(S.F.γt.K3, symmetries_K3a)
end


function zero_out_points_outside_box!(
    S :: NL_ParquetSolver
)
    mK_Γ = get_P_mesh(S.F)
    symmetries_K1p = [Symmetry{2}(w -> sK1_conj(w, mK_Γ)),]
    symmetries_K2p = [Symmetry{3}(w -> sK2pp1(w, mK_Γ)),
                      Symmetry{3}(w -> sK2pp2(w, mK_Γ)),]
    symmetries_K3p = [Symmetry{4}(w -> sK3pp1(w, mK_Γ)),
                      Symmetry{4}(w -> sK3pp2(w, mK_Γ)),
                      Symmetry{4}(w -> sK3pp3(w, mK_Γ)),]
    symmetries_K1a = [Symmetry{2}(w -> sK1_conj(w, mK_Γ)),]
    symmetries_K2a = [Symmetry{3}(w -> sK2ph1(w, mK_Γ)),
                      Symmetry{3}(w -> sK2ph2(w, mK_Γ)),]
    symmetries_K3a = [Symmetry{4}(w -> sK3ph1(w, mK_Γ)),
                      Symmetry{4}(w -> sK3ph2(w, mK_Γ)),
                      Symmetry{4}(w -> sK3ph3(w, mK_Γ)),]

    zero_out_points_outside_box!(S.F.γp.K1, symmetries_K1p)
    zero_out_points_outside_box!(S.F.γp.K2, symmetries_K2p)
    zero_out_points_outside_box!(S.F.γp.K3, symmetries_K3p)
    zero_out_points_outside_box!(S.F.γa.K1, symmetries_K1a)
    zero_out_points_outside_box!(S.F.γa.K2, symmetries_K2a)
    zero_out_points_outside_box!(S.F.γa.K3, symmetries_K3a)
    zero_out_points_outside_box!(S.F.γt.K1, symmetries_K1a)
    zero_out_points_outside_box!(S.F.γt.K2, symmetries_K2a)
    zero_out_points_outside_box!(S.F.γt.K3, symmetries_K3a)
end


function zero_out_points_outside_box!(
    S :: ParquetSolver
)
    symmetries_K1p = [Symmetry{1}(sK1pp)]
    symmetries_K2p = [Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)]
    symmetries_K3p = [Symmetry{3}(sK3pp1), Symmetry{3}(sK3pp2), Symmetry{3}(sK3pp3)]
    symmetries_K1a = [Symmetry{1}(sK1ph)]
    symmetries_K2a = [Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)]
    symmetries_K3a = [Symmetry{3}(sK3ph1), Symmetry{3}(sK3ph2), Symmetry{3}(sK3ph3)]

    zero_out_points_outside_box!(S.F.γp.K1, symmetries_K1p)
    zero_out_points_outside_box!(S.F.γp.K2, symmetries_K2p)
    zero_out_points_outside_box!(S.F.γp.K3, symmetries_K3p)
    zero_out_points_outside_box!(S.F.γa.K1, symmetries_K1a)
    zero_out_points_outside_box!(S.F.γa.K2, symmetries_K2a)
    zero_out_points_outside_box!(S.F.γa.K3, symmetries_K3a)
    zero_out_points_outside_box!(S.F.γt.K1, symmetries_K1a)
    zero_out_points_outside_box!(S.F.γt.K2, symmetries_K2a)
    zero_out_points_outside_box!(S.F.γt.K3, symmetries_K3a)
end
