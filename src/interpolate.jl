function _fourier_interpolate!(yi :: AbstractVector, Lo, Li)
    # Fourier interpolate data of size Li^2 to data of size Lo^2
    @assert size(yi) == (Li^2,)
    yo = zeros(eltype(yi), Lo^2)

    yi_R = fft(reshape(yi, Li, Li)) ./ Li^2
    yo_R = Base.ReshapedArray(yo, (Lo, Lo), ())

    Rs_1d = (-div(Li, 2)) : div(Li, 2)
    Rs = collect(Iterators.product(Rs_1d, Rs_1d))
    for R in Rs
        weight = 1.0
        if mod(Li, 2) == 0
            abs(R[1]) == div(Li, 2) && (weight /= 2)
            abs(R[2]) == div(Li, 2) && (weight /= 2)
        end

        # Index of R in Ki and Ko
        iRi = mod.(R, (Li, Li)) .+ 1
        iRo = mod.(R, (Lo, Lo)) .+ 1

        yo_R[iRo...] += yi_R[iRi...] .* weight
    end
    bfft!(yo_R)

    return yo
end

function _fourier_interpolate!(yi :: AbstractMatrix, Lo, Li)
    # Fourier interpolate data of size (Li^2, Li^2) to data of size (Lo^2, Lo^2)
    @assert size(yi) == (Li^2, Li^2)
    yo = zeros(eltype(yi), Lo^2, Lo^2)

    yi_R = fft(reshape(yi, Li, Li, Li, Li)) ./ Li^4
    yo_R = Base.ReshapedArray(yo, (Lo, Lo, Lo, Lo), ())

    Rs_1d = (-div(Li, 2)) : div(Li, 2)
    Rs = collect(Iterators.product(Rs_1d, Rs_1d))
    for R2 in Rs, R1 in Rs
        weight = 1.0
        if mod(Li, 2) == 0
            abs(R1[1]) == div(Li, 2) && (weight /= 2)
            abs(R1[2]) == div(Li, 2) && (weight /= 2)
            abs(R2[1]) == div(Li, 2) && (weight /= 2)
            abs(R2[2]) == div(Li, 2) && (weight /= 2)
        end

        # Index of R in Ki and Ko
        iRi1 = mod.(R1, (Li, Li)) .+ 1
        iRi2 = mod.(R2, (Li, Li)) .+ 1
        iRo1 = mod.(R1, (Lo, Lo)) .+ 1
        iRo2 = mod.(R2, (Lo, Lo)) .+ 1

        yo_R[iRo1..., iRo2...] += yi_R[iRi1..., iRi2...] .* weight
    end

    bfft!(yo_R)

    return yo
end

function interpolate_vertex!(Ko :: T, Ki :: T) where {T <: Union{NL_MF_K1, NL_MF_G}}
    # Fourier interpolation
    set!(Ko, 0)

    Lo = bz(meshes(Ko, Val(2))).L
    Li = bz(meshes(Ki, Val(2))).L

    for i in eachindex(Ko.meshes[1])
        Ω = value(Ko.meshes[1][i])
        is_inbounds(Ω, Ki.meshes[1]) || continue

        i_ = MatsubaraFunctions.mesh_index(Ω, Ki.meshes[1])

        Ko.data[i, :] .= _fourier_interpolate!(Ki.data[i_, :], Lo, Li)
    end

    return nothing
end

function interpolate_vertex!(Ko :: NL_MF_K2, Ki :: NL_MF_K2)
    set!(Ko, 0)

    Lo = bz(meshes(Ko, Val(3))).L
    Li = bz(meshes(Ki, Val(3))).L

    for i2 in eachindex(Ko.meshes[2]), i1 in eachindex(Ko.meshes[1])
        ω1 = value(Ko.meshes[1][i1])
        ω2 = value(Ko.meshes[2][i2])
        is_inbounds(ω1, Ki.meshes[1]) || continue
        is_inbounds(ω2, Ki.meshes[2]) || continue

        i1_ = MatsubaraFunctions.mesh_index(ω1, Ki.meshes[1])
        i2_ = MatsubaraFunctions.mesh_index(ω2, Ki.meshes[2])

        Ko.data[i1, i2, :] .= _fourier_interpolate!(Ki.data[i1_, i2_, :], Lo, Li)
    end

    return nothing
end

function interpolate_vertex!(Ko :: NL_MF_K3, Ki :: NL_MF_K3)
    set!(Ko, 0)

    Lo = bz(meshes(Ko, Val(4))).L
    Li = bz(meshes(Ki, Val(4))).L

    for i3 in eachindex(Ko.meshes[3]), i2 in eachindex(Ko.meshes[2]), i1 in eachindex(Ko.meshes[1])
        ω1 = value(Ko.meshes[1][i1])
        ω2 = value(Ko.meshes[2][i2])
        ω3 = value(Ko.meshes[3][i3])
        is_inbounds(ω1, Ki.meshes[1]) || continue
        is_inbounds(ω2, Ki.meshes[2]) || continue
        is_inbounds(ω3, Ki.meshes[3]) || continue

        i1_ = MatsubaraFunctions.mesh_index(ω1, Ki.meshes[1])
        i2_ = MatsubaraFunctions.mesh_index(ω2, Ki.meshes[2])
        i3_ = MatsubaraFunctions.mesh_index(ω3, Ki.meshes[3])

        Ko.data[i1, i2, i3, :] .= _fourier_interpolate!(Ki.data[i1_, i2_, i3_, :], Lo, Li)
    end

    return nothing
end

function interpolate_vertex!(Ko :: NL2_MF_K2, Ki :: NL2_MF_K2)
    set!(Ko, 0)

    Lo = bz(meshes(Ko, Val(3))).L
    Li = bz(meshes(Ki, Val(3))).L

    for i2 in eachindex(Ko.meshes[2]), i1 in eachindex(Ko.meshes[1])
        ω1 = value(Ko.meshes[1][i1])
        ω2 = value(Ko.meshes[2][i2])
        is_inbounds(ω1, Ki.meshes[1]) || continue
        is_inbounds(ω2, Ki.meshes[2]) || continue

        i1_ = MatsubaraFunctions.mesh_index(ω1, Ki.meshes[1])
        i2_ = MatsubaraFunctions.mesh_index(ω2, Ki.meshes[2])

        Ko.data[i1, i2, :, :] .= _fourier_interpolate!(Ki.data[i1_, i2_, :, :], Lo, Li)
    end

    return nothing
end

function interpolate_solver!(
    So :: ST,
    Si :: ST;
    occ_target = nothing,
    hubbard_params = nothing
    ) where {ST <: AbstractSolver}

    # Interpolate the self-energy and vertex of Si to the mesh of So

    # Interpolate self-energy wavevector mesh
    Σi = MeshFunction(meshes(Si.Σ, Val(1)), meshes(So.Σ, Val(2)); data_t = eltype(Si.Σ.data))
    interpolate_vertex!(Σi, Si.Σ)

    @assert meshes(So.Σ, Val(2)) == meshes(Σi, Val(2))
    for i in eachindex(So.Σ.data)
        ν, k = value.(to_meshes(So.Σ, i))
        if value(ν) < plain_value(meshes(Σi, Val(1))[1])
            So.Σ[ν, k] = Σi(value(meshes(Σi, Val(1))[1]), k)
        elseif value(ν) > plain_value(meshes(Σi, Val(1))[end])
            So.Σ[ν, k] = Σi(value(meshes(Σi, Val(1))[end]), k)
        else
            So.Σ[ν, k] = Σi(ν, k)
        end
    end

    if occ_target !== nothing
        # Update chemical potential to fix the occupation
        μ = compute_hubbard_chemical_potential(occ_target, So.Σ, hubbard_params)
        set!(So.Gbare, hubbard_bare_Green(meshes(So.Gbare)...; μ, hubbard_params...))
    end
    Dyson!(So)
    bubbles!(So)

    for Ch in (aCh, pCh, tCh)
        γo = fdDGAsolver.get_reducible_vertex(So.F, Ch)
        γi = fdDGAsolver.get_reducible_vertex(Si.F, Ch)
        interpolate_vertex!(γo.K1, γi.K1)
        interpolate_vertex!(γo.K2, γi.K2)
        interpolate_vertex!(γo.K3, γi.K3)
    end

    symmetrize_solver!(So)

    return nothing
end
