# Precompute the vertex components that are used repeatedly in the BSE for the K3 vertex.
#
# The K3 channel BSE has the form ``C(Ω, ν, ν') = A(Ω, ν, ω) * Π(Ω, ω) * B(Ω, ω, ν')``.
# Since the vertex `A(Ω, ν, ω)` does not depend on `ν'`, it can be computed outside the
# loop over `ν'`. The same applies to the vertex `B(Ω, ω, ν')`.
# Here, we compute all the necessary vertices that appears in the BSE for the K3 vertex.
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
        Ω, ω, νp, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        S.cache_Γpx[i] = S.F( Ω, ω, νp, P, kSW, kSW, pCh, xSp; F0=false, γp=false)

        S.cache_F0p[i] = ( S.F0(Ω, ω,   νp, P, kSW, kSW, pCh, xSp)
                         - S.F0(Ω, ω, νInf, P, kSW, kSW, pCh, xSp) )
        S.cache_F0a[i] = ( S.F0(Ω, ω,   νp, P, kSW, kSW, aCh, pSp)
                         - S.F0(Ω, ω, νInf, P, kSW, kSW, aCh, pSp) )
        S.cache_F0t[i] = ( S.F0(Ω, ω,   νp, P, kSW, kSW, tCh, pSp)
                         - S.F0(Ω, ω, νInf, P, kSW, kSW, tCh, pSp) )

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
        Ω, ν, ω, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # r-irreducible vertex in each channel r = p, a, t
        S.cache_Γpp[i] = S.F(Ω, ν, ω, P, kSW, kSW, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, P, kSW, kSW, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, P, kSW, kSW, tCh, pSp; F0=false, γt=false)

        # Total vertex in each channel. We compute the reducible part and add the
        # irreducible part computed above
        S.cache_Fp[i]  = ( S.F(Ω,    ν, ω, P, kSW, kSW, pCh, pSp; γa=false, γt=false)
                         - S.F(Ω, νInf, ω, P, kSW, kSW, pCh, pSp; γa=false, γt=false)
                         + S.cache_Γpp[i] )
        S.cache_Fa[i]  = ( S.F(Ω,    ν, ω, P, kSW, kSW, aCh, pSp; γp=false, γt=false)
                         - S.F(Ω, νInf, ω, P, kSW, kSW, aCh, pSp; γp=false, γt=false)
                         + S.cache_Γa[i] )
        S.cache_Ft[i]  = ( S.F(Ω,    ν, ω, P, kSW, kSW, tCh, pSp; γp=false, γa=false)
                         - S.F(Ω, νInf, ω, P, kSW, kSW, tCh, pSp; γp=false, γa=false)
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


function build_K3_cache_mfRG!(
    S :: NL2_ParquetSolver{Q},
    is_first_iteration :: Bool,
    ) where {Q}

    # If `is_first_iteration = true`, compute F0 and Γ.
    # If `is_first_iteration = false`, compute Γ.

    # The case (3) `cache_F*` is now computed with the reference vertex, not the target vertex.

    # Vertices multiplied by bubbles from the left
    @inline function diagram_Γpx(wtpl)
        Ω, ω, νp, P = wtpl
        return S.F(Ω, ω, νp, P, kSW, kSW, pCh, xSp; F0=false, γp=false)
    end
    S.SGpp[3](S.cache_Γpx, InitFunction{4, Q}(diagram_Γpx); mode = S.mode)


    # Vertices multiplied by bubbles from the right
    @inline function diagram_Γpp(wtpl)
        Ω, ν, ω, P = wtpl
        return S.F(Ω, ν, ω, P, kSW, kSW, pCh, pSp; F0=false, γp=false)
    end
    @inline function diagram_Γa(wtpl)
        Ω, ν, ω, P = wtpl
        return S.F(Ω, ν, ω, P, kSW, kSW, aCh, pSp; F0=false, γa=false)
    end
    @inline function diagram_Γt(wtpl)
        Ω, ν, ω, P = wtpl
        return S.F(Ω, ν, ω, P, kSW, kSW, tCh, pSp; F0=false, γt=false)
    end

    S.SGpp[3](S.cache_Γpp, InitFunction{4, Q}(diagram_Γpp); mode = S.mode)
    S.SGph[3](S.cache_Γa,  InitFunction{4, Q}(diagram_Γa); mode = S.mode)
    S.SGph[3](S.cache_Γt,  InitFunction{4, Q}(diagram_Γt); mode = S.mode)

    # Convert from pSp (parallel spin component) to dSp (density component)
    # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
    @. S.cache_Γt.data = S.cache_Γt.data * 2 - S.cache_Γa.data


    if is_first_iteration
        # Reference vertex. Needs to be cached only in the first mfRG iteration.

        @inline function diagram_F0p(wtpl)
            Ω, ν, ω, P = wtpl
            return S.F0(Ω, ν, ω, P, kSW, kSW, pCh, pSp) - S.F0(Ω, νInf, ω, P, kSW, kSW, pCh, pSp)
        end
        @inline function diagram_F0a(wtpl)
            Ω, ν, ω, P = wtpl
            return S.F0(Ω, ν, ω, P, kSW, kSW, aCh, pSp) - S.F0(Ω, νInf, ω, P, kSW, kSW, aCh, pSp)
        end
        @inline function diagram_F0t(wtpl)
            Ω, ν, ω, P = wtpl
            return S.F0(Ω, ν, ω, P, kSW, kSW, tCh, pSp) - S.F0(Ω, νInf, ω, P, kSW, kSW, tCh, pSp)
        end

        S.SGpp[3](S.cache_Fp, InitFunction{4, Q}(diagram_F0p); mode = S.mode)
        S.SGph[3](S.cache_Fa, InitFunction{4, Q}(diagram_F0a); mode = S.mode)
        S.SGph[3](S.cache_Ft, InitFunction{4, Q}(diagram_F0t); mode = S.mode)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        @. S.cache_Ft.data = S.cache_Ft.data * 2 - S.cache_Fa.data
    end

    return nothing
end
