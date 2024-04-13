# Build cache MeshFunction for K3 BSE
# One of the fermionic frequencies and momenta is multiplied to the bubble, so we use the
# mesh of the bubble as the mesh of the cached vertex.
function build_K3_cache(
    S :: NL2_ParquetSolver{Q}
    ) where {Q}

    mΠν = meshes(S.Πph, Val(2))
    mΠk = meshes(S.Πph, Val(4))

    # Vertices multiplied by bubbles from the left

    Γpx = MeshFunction(meshes(S.F.γp.K3, Val(1)), mΠν, meshes(S.F.γp.K3, Val(3)), meshes(S.F.γp.K3, Val(4)), mΠk; data_t=Q)
    F0p = copy(Γpx)
    F0a = copy(Γpx)
    F0t = copy(Γpx)

    Threads.@threads for i in CartesianIndices(Γpx.data)
        Ω  = value(meshes(Γpx, Val(1))[i.I[1]])
        ν  = value(meshes(Γpx, Val(2))[i.I[2]])
        νp = value(meshes(Γpx, Val(3))[i.I[3]])
        P  = value(meshes(Γpx, Val(4))[i.I[4]])
        k  = value(meshes(Γpx, Val(5))[i.I[5]])
        Γpx[i] = S.F( Ω, ν, νp, P, k, kSW, pCh, xSp; F0=false, γp=false)
        F0p[i] = S.F0(Ω, ν, νp, P, k, kSW, pCh, xSp)
        F0a[i] = S.F0(Ω, ν, νp, P, k, kSW, aCh, pSp)
        F0t[i] = S.F0(Ω, ν, νp, P, k, kSW, tCh, pSp)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        F0t[i] = 2 * F0t[i] - F0a[i]
    end

    # Vertices multiplied by bubbles from the right

    Γpp = MeshFunction(meshes(S.F.γp.K3, Val(1)), meshes(S.F.γp.K3, Val(2)), mΠν, meshes(S.F.γp.K3, Val(4)), mΠk; data_t=Q)
    Γa = deepcopy(Γpp)
    Γt = deepcopy(Γpp)
    Fp = deepcopy(Γpp)
    Fa = deepcopy(Γpp)
    Ft = deepcopy(Γpp)

    Threads.@threads for i in CartesianIndices(Γpp.data)
        Ω  = value(meshes(Γpp, Val(1))[i.I[1]])
        ν  = value(meshes(Γpp, Val(2))[i.I[2]])
        νp = value(meshes(Γpp, Val(3))[i.I[3]])
        P  = value(meshes(Γpp, Val(4))[i.I[4]])
        kp = value(meshes(Γpp, Val(5))[i.I[5]])

        # r-irreducible vertex in each channel r = p, a, t
        Γpp[i] = S.F(Ω, ν, νp, P, kSW, kp, pCh, pSp; F0=false, γp=false)
        Γa[i]  = S.F(Ω, ν, νp, P, kSW, kp, aCh, pSp; F0=false, γa=false)
        Γt[i]  = S.F(Ω, ν, νp, P, kSW, kp, tCh, pSp; F0=false, γt=false)

        # Total vertex in each channel. We compute the reducible part and add the
        # irreducible part computed above
        Fp[i]  = S.F(Ω, ν, νp, P, kSW, kp, pCh, pSp; γa=false, γt=false) + Γpp[i]
        Fa[i]  = S.F(Ω, ν, νp, P, kSW, kp, aCh, pSp; γp=false, γt=false) + Γa[i]
        Ft[i]  = S.F(Ω, ν, νp, P, kSW, kp, tCh, pSp; γp=false, γa=false) + Γt[i]

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        Γt[i] = Γt[i] * 2 - Γa[i]
        Ft[i] = Ft[i] * 2 - Fa[i]
    end


    (; Γpx, F0p, F0a, F0t, Γpp, Γa, Γt, Fp, Fa, Ft)
end
