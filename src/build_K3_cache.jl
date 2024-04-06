# Build cache MeshFunction for K3 BSE
function build_K3_cache(S :: ParquetSolver{Q}) where {Q}

    mΠν = meshes(S.Πph, 2)

    # Vertices multiplied by bubbles to the left

    # Γpx : Target   , p-channel irreducible, spin cross
    # F0p : Reference, p-channel full       , spin cross
    # F0a : Reference, a-channel full       , spin parallel
    # F0t : Reference, t-channel full       , spin density
    Γpx = MeshFunction(meshes(S.F.γp.K3, 1), mΠν, meshes(S.F.γp.K3, 2); data_t = Q)
    F0p = copy(Γpx)
    F0a = copy(Γpx)
    F0t = copy(Γpx)

    Threads.@threads for i in CartesianIndices(Γpx.data)
        Ω  = value(meshes(Γpx, 1)[i.I[1]])
        ν  = value(meshes(Γpx, 2)[i.I[2]])
        νp = value(meshes(Γpx, 3)[i.I[3]])
        Γpx[i] = S.F(Ω, ν, νp, pCh, xSp; F0=false, γp=false)
        F0p[i] = S.F0(Ω, ν, νp, pCh, xSp)
        F0a[i] = S.F0(Ω, ν, νp, aCh, pSp)
        F0t[i] = S.F0(Ω, ν, νp, tCh, pSp) * 2 - F0a[i]
    end

    # Vertices multiplied by bubbles from the right

    # Γpp : Target, p-channel irreducible, spin parallel
    # Γa  : Target, a-channel irreducible, spin parallel
    # Γt  : Target, t-channel irreducible, spin density
    # Fp  : Target, a-channel full,        spin parallel
    # Fa  : Target, a-channel full,        spin parallel
    # Ft  : Target, a-channel full,        spin density
    Γpp = MeshFunction(meshes(S.F.γp.K3, 1), meshes(S.F.γp.K3, 2), mΠν; data_t = Q)
    Γa = deepcopy(Γpp)
    Γt = deepcopy(Γpp)
    Fp = deepcopy(Γpp)
    Fa = deepcopy(Γpp)
    Ft = deepcopy(Γpp)

    Threads.@threads for i in CartesianIndices(Γpp.data)
        Ω  = value(meshes(Γpp, 1)[i.I[1]])
        ν  = value(meshes(Γpp, 2)[i.I[2]])
        νp = value(meshes(Γpp, 3)[i.I[3]])
        Γpp[i] = S.F(Ω, ν, νp, pCh, pSp; F0=false, γp=false)
        Γa[i] = S.F(Ω, ν, νp, aCh, pSp; F0=false, γa=false)
        Γt[i] = S.F(Ω, ν, νp, tCh, pSp; F0=false, γt=false) * 2 - Γa[i]
        Fp[i] = S.F(Ω, ν, νp, pCh, pSp)
        Fa[i] = S.F(Ω, ν, νp, aCh, pSp)
        Ft[i] = S.F(Ω, ν, νp, tCh, pSp) * 2 - Fa[i]
    end

    (; Γpx, F0p, F0a, F0t, Γpp, Γa, Γt, Fp, Fa, Ft)
end
