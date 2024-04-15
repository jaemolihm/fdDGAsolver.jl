# Precompute the vertex components that are used repeatedly in the BSE for the K3 vertex.
#
# The K3 channel BSE has the form ``C(Ω, ν, ν') = A(Ω, ν, ω) * Π(Ω, ω) * B(Ω, ω, ν')``.
# Since the vertex `A(Ω, ν, ω)` does not depend on `ν'`, it can be computed outside the
# loop over `ν'`. The same applies to the vertex `B(Ω, ω, ν')`.
# Here, we compute all the necessary vertices that appears in the BSE for the K3 vertex.
# In the code, we denote the inner, integrated frequency variable with `ω` and the
# outer frequency variable with `ν` and `νp`.
#
# The `ω` component of the precomputed vertex should be computed on the frequency mesh of
# the bubble, which is larger than that of the K3 vertex.
#
# We store only the decaying contributions of the vertices, i.e., we subtract the asymptotic
# contributions, as they are accounted for in the BSE for the K1 and K2 vertices.
#
# There are multiple types of vertices to precompute.
# (1) `cache_F0*`: Full vertex of the reference system (`S.F0`)
# (2) `cache_Γ*` : Irreducible finite-difference vertex (`S.F` with `F0=false`, `γ*=false`)
# (3) `cache_F*` : Full vertex of the target system (`S.F`)
function build_K3_cache!(
    S :: ParquetSolver{Q}
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

    # Vertices multiplied by bubbles to the left (by ω)
    # Γpx : Target, irreducible vertex in the p channel, xSp component.

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ω, νp = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        S.cache_Γpx[i] = S.F( Ω, ω, νp, pCh, xSp; F0=false, γp=false)
        S.cache_F0p[i] = S.F0(Ω, ω, νp, pCh, xSp) - S.F0(Ω, ω, νInf, pCh, xSp)
        S.cache_F0a[i] = S.F0(Ω, ω, νp, aCh, pSp) - S.F0(Ω, ω, νInf, aCh, pSp)
        S.cache_F0t[i] = S.F0(Ω, ω, νp, tCh, pSp) - S.F0(Ω, ω, νInf, tCh, pSp)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # Vertices multiplied by bubbles from the right (by ω)

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, ω = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # r-irreducible vertex in each channel r = p, a, t
        S.cache_Γpp[i] = S.F(Ω, ν, ω, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, tCh, pSp; F0=false, γt=false)

        # Total vertex in each channel. We compute the reducible part and add the
        # irreducible part computed above
        S.cache_Fp[i]  = ( S.F(Ω,    ν, ω, pCh, pSp; γa=false, γt=false)
                         - S.F(Ω, νInf, ω, pCh, pSp; γa=false, γt=false)
                         + S.cache_Γpp[i] )

        S.cache_Fa[i]  = ( S.F(Ω,    ν, ω, aCh, pSp; γp=false, γt=false)
                         - S.F(Ω, νInf, ω, aCh, pSp; γp=false, γt=false)
                         + S.cache_Γa[i] )

        S.cache_Ft[i]  = ( S.F(Ω,    ν, ω, tCh, pSp; γp=false, γa=false)
                         - S.F(Ω, νInf, ω, tCh, pSp; γp=false, γa=false)
                         + S.cache_Γt[i] )

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
