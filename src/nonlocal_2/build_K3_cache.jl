# Build cache MeshFunction for K3 BSE
# One of the fermionic frequencies and momenta is multiplied to the bubble, so we use the
# mesh of the bubble as the mesh of the cached vertex.
function build_K3_cache!(
    S :: NL2_ParquetSolver{Q}
    ) where {Q}

    set!(S.cache_Γpx, 0)
    set!(S.cache_F0p, 0)
    set!(S.cache_F0a, 0)
    set!(S.cache_F0t, 0)
    set!(S.cache_Γpp, 0)
    set!(S.cache_Γa,  0)
    set!(S.cache_Γt,  0)
    set!(S.cache_Fp,  0)
    set!(S.cache_Fa,  0)
    set!(S.cache_Ft,  0)

    # Vertices multiplied by bubbles from the left

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ν, νp, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        S.cache_Γpx[i] = S.F( Ω, ν, νp, P, kSW, kSW, pCh, xSp; F0=false, γp=false)
        S.cache_F0p[i] = S.F0(Ω, ν, νp, P, kSW, kSW, pCh, xSp)
        S.cache_F0a[i] = S.F0(Ω, ν, νp, P, kSW, kSW, aCh, pSp)
        S.cache_F0t[i] = S.F0(Ω, ν, νp, P, kSW, kSW, tCh, pSp)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # Vertices multiplied by bubbles from the right

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, νp, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # r-irreducible vertex in each channel r = p, a, t
        S.cache_Γpp[i] = S.F(Ω, ν, νp, P, kSW, kSW, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, νp, P, kSW, kSW, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, νp, P, kSW, kSW, tCh, pSp; F0=false, γt=false)

        # Total vertex in each channel. We compute the reducible part and add the
        # irreducible part computed above
        S.cache_Fp[i]  = S.F(Ω, ν, νp, P, kSW, kSW, pCh, pSp; γa=false, γt=false) + S.cache_Γpp[i]
        S.cache_Fa[i]  = S.F(Ω, ν, νp, P, kSW, kSW, aCh, pSp; γp=false, γt=false) + S.cache_Γa[i]
        S.cache_Ft[i]  = S.F(Ω, ν, νp, P, kSW, kSW, tCh, pSp; γp=false, γa=false) + S.cache_Γt[i]

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_Γt[i] = S.cache_Γt[i] * 2 - S.cache_Γa[i]
        S.cache_Ft[i] = S.cache_Ft[i] * 2 - S.cache_Fa[i]
    end

    mpi_allreduce!(S.cache_Γpp)
    mpi_allreduce!(S.cache_Γa)
    mpi_allreduce!(S.cache_Γt)
    mpi_allreduce!(S.cache_Fp)
    mpi_allreduce!(S.cache_Fa)
    mpi_allreduce!(S.cache_Ft)

    return nothing
end
