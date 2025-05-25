# Visualize local vertex.
# Not included in the package to avoid having PyPlot (or Requires) as dependency.
# To use these functions, `include` the source file in your script.

function plot_self_energy(Σ, ax1 = gca(), ax2 = gca(); show = false, kwargs...)
    ax1.plot(values(Σ.meshes[1]), real.(Σ.data .* -im), "o-"; kwargs...)
    ax2.plot(values(Σ.meshes[1]), imag.(Σ.data .* -im), "x-"; kwargs...)
    if show
        ax1.set_xlim([-10, 10] .* (2π * temperature(Γ)))
        ax2.set_xlim([-10, 10] .* (2π * temperature(Γ)))
        ax1.legend()
        fig = gcf(); display(fig); close(fig)
    end
    return nothing
end


function plot_vertex_K1(Γ)
    # Plot K1 vertex
    for (K1, label) in zip([Γ.γp.K1, Γ.γt.K1, Γ.γa.K1], ["K1p", "K1t", "K1a"])
        plot(values(K1.meshes[1]), real.(K1.data), "o-"; label)
    end
    xlim([-10, 10] .* (2π * temperature(Γ)))
    legend()
    fig = gcf(); display(fig); close(fig)
end


function plot_vertex_K2(Γ; vmax = 1.0)
    # Plot K2 vertex

    fig, plotaxes = subplots(3, 2, figsize=(6, 9); sharex=true, sharey=true)

    extent = [extrema(values(meshes(Γ.γa.K2, Val(1))))..., extrema(values(meshes(Γ.γa.K2, Val(2))))...]

    kwargs = (; vmin=-vmax, vmax, cmap="RdBu_r", aspect="auto", interpolation="none", extent, origin="lower")

    for (i, (K2, label)) in enumerate(zip([Γ.γp.K2, Γ.γt.K2, Γ.γa.K2], ["K2p", "K2t", "K2a"]))
        plotaxes[i, 1].imshow(real.(K2.data); kwargs...)
        plotaxes[i, 2].imshow(imag.(K2.data); kwargs...)
        plotaxes[i, 1].set_title("Re $label (Ω, ν)")
        plotaxes[i, 2].set_title("Im $label (Ω, ν)")
    end

    for ax in plotaxes
        ax.set_xlabel("ν")
        ax.set_ylabel("Ω")
        # ax.set_xlim([-40, 40] .* (2π * temperature(Γ)))
        # ax.set_ylim([-40, 40] .* (2π * temperature(Γ)))
    end

    display(fig); close(fig)
end

function plot_vertex_core(Γ; vmax = 0.1, Ω = 0.0)
    # Plot core vertex

    fig, plotaxes = subplots(4, 2, figsize=(6, 12); sharex=true, sharey=true)

    mesh_Ω = meshes(Γ.F0.Ft_p, Val(1))
    iΩ = MatsubaraFunctions.mesh_index(Ω, mesh_Ω)

    for (i, (Λ, label)) in enumerate(zip([Γ.F0.Ft_p, Γ.F0.Ft_x, Γ.F0.Fp_p, Γ.F0.Fp_x], ["Ft_p", "Ft_x", "Fp_p", "Fp_x"]))
        plotaxes[i, 1].imshow(real.(Λ.data[iΩ,:,:])'; vmin=-vmax, vmax, cmap="RdBu_r", aspect="auto", interpolation="none")
        plotaxes[i, 2].imshow(imag.(Λ.data[iΩ,:,:])'; vmin=-vmax, vmax, cmap="RdBu_r", aspect="auto", interpolation="none")

        plotaxes[i, 1].set_title("Re $label (Ω, ν, ν')")
        plotaxes[i, 2].set_title("Im $label (Ω, ν, ν')")
    end

    for ax in plotaxes
        ax.set_xlabel("ν")
        ax.set_ylabel("ν'")
    end

    fig.suptitle("Ω = $(plain_value(mesh_Ω[iΩ]))")
    tight_layout()

    display(fig); close(fig)
end
