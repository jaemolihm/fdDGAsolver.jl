# compute tail moments in cubic approximation from upper bound of 1D MatsubaraFunction
# Note: the distance between interpolation nodes is kept constant below T = 1
"""
    function upper_tail_moments(
        f  :: MF_G{Q},
        α0 :: Number,
        )  :: Tuple{Q, Q} where {Q}

Returns high frequency moments for cubic model using upper grid bound.
`α0` is the asymptotic limit for large positive frequencies.
"""
function upper_tail_moments(
    f  :: MF_G{Q},
    α0 :: Number,
    )  :: Vector{Q} where {Q}

    # compute interpolation nodes
    n = length(meshes(f, 1))
    dist = ceil(Int64, 1 / min(temperature(meshes(f, 1)), 1))
    if (n - 2dist) < ceil(Int64, 0.75 * n)
        throw(ArgumentError("Grid is too small for extrapolation"))
    end

    # Fit to y(x) = A₁ / x + A₂ / x² + A₃ / x³
    inds = [n, n - dist, n - 2dist]
    y = [f[i] - Q(α0) for i in inds]
    x = [plain_value(meshes(f, 1)[i]) for i in inds]
    return [1 ./ x ;; 1 ./ x.^2] \ y
end

# compute tail moments in cubic approximation from lower bound of 1D MatsubaraFunction
# Note: the distance between interpolation nodes is kept constant below T = 1
"""
    function lower_tail_moments(
        f  :: MF_G{Q},
        α0 :: Number,
        )  :: Tuple{Q, Q} where {Q}

Returns high frequency moments for quadratic model using lower grid bound.
`α0` is the asymptotic limit for large negative frequencies.
"""
function lower_tail_moments(
    f  :: MF_G{Q},
    α0 :: Number,
    )  :: Vector{Q} where {Q}

    # compute interpolation nodes
    n = length(meshes(f, 1))
    dist = ceil(Int64, 1 / min(temperature(meshes(f, 1)), 1))
    if 1 + dist > floor(Int64, 0.25 * n)
        throw(ArgumentError("Grid is too small for extrapolation"))
    end

    # Fit to y(x) = A₁ / x + A₂ / x² + A₃ / x³
    inds = [1, 1 + dist, 1 + 2dist]
    y = [f[i] - Q(α0) for i in inds]
    x = [plain_value(meshes(f, 1)[i]) for i in inds]
    return [1 ./ x ;; 1 ./ x.^2] \ y
end

"""
    function sum_me(
        f :: MatsubaraFunction{1, SD, DD, Q},
        x :: Vararg{Int64, SD}
        ) :: Q where {SD, DD, Q <: Complex}

Computes the fermionic Matsubara sum (with regulator exp(-iw0+)) for a complex valued MatsubaraFunction on a 1D grid. This is only viable if `f` has
a Laurent series representation with respect to an annulus about the imaginary axis and decays to zero.
"""
function sum_me(
    G :: MF_G{Q},
    ) :: Q where {Q}

    # sanity check for current implementation, lift this restriction as soon as possible
    if !(meshes(G, 1) isa FMesh)
        throw(ArgumentError("Extrapolation is currently limited to fermionic meshes"))
    end

    # compute tail moments
    upper_moments = upper_tail_moments(G, Q(0))
    lower_moments = lower_tail_moments(G, Q(0))
    upper_max = max(abs.(upper_moments)...)
    lower_max = max(abs.(lower_moments)...)

    # check self-consistency
    diff  = max(abs.(upper_moments .- lower_moments)...);
    scale = 1e-3 + max(upper_max, lower_max) * 1e-2;
    err   = diff / scale

    if err >= 1
        @warn "Tail fits are inconsistent: upper_moments = $(upper_moments), lower_moments = $(lower_moments)"
    end

    # compute expansion coefficients
    α1 =  (upper_moments[1] + lower_moments[1]) / 2 * im
    α2 = -(upper_moments[2] + lower_moments[2]) / 2

    # compute the Matsubara sum using quadratic asymptotic model
    T   = temperature(meshes(G, 1))
    val = T * sum(view(G, :)) - (α1 + α2 / T / 2) / 2

    for i in eachindex(meshes(G, 1))
        val += T * α2 / plain_value(meshes(G, 1)[i]) / plain_value(meshes(G, 1)[i])
    end

    return val
end
# 1 / (maxv)
