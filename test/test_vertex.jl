using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Test

@testset "Vertex" begin
    T = 1.0
    N = 10
    mesh = MatsubaraMesh(T, N, Fermion)
    x = MeshFunction((mesh,), rand(length(mesh)))
    y = Vertex_K1(x, :a, 3.)
    @test y[mesh[1]] === x[mesh[1]]
    @test Base.show(IOBuffer(), y) === nothing
    @test fdDGAsolver.channel_freq(y) == :a

    v1, v2, w = -10., 9., 5.
    @test y(v1, v2, w) == x(w)
    @test y(v1, v2, w, :a) == x(w)
    @test y(v1, v2, w, :p) == x(-v1 - v2 - w)
    @test y(v1, v2, w, :t) == x(v1 - v2)
    @test y(v1, v2, 10_000.) == 3.

    n = length(mesh)
    x_K2 = MeshFunction((mesh, mesh), rand(n, n))
    x_K3 = MeshFunction((mesh, mesh, mesh), rand(n, n, n))
    y_K2  = Vertex_K2( x_K2, :a, 5.)
    y_K2p = Vertex_K2p(x_K2, :a, 6.)
    y_K3  = Vertex_K3( x_K3, :a, 7.)
    @test size(y_K2)  === (n, n)
    @test size(y_K2p) === (n, n)
    @test size(y_K3)  === (n, n, n)

    v1, v2, w = 0., 1., 3.
    @test y_K2.data === y_K2p.data
    @test y_K2(v1, v2, w) ==  y_K2p(v2, v1, w)
    @test y_K2(10_000., v1, w) == 5.0
    @test y_K2p(v1, 10_000., w) == 6.0
    @test y_K2(v1, 10_000., w) == x_K2(v1, w)
    @test y_K2p(10_000., v1, w) == x_K2(v1, w)
    @test y_K2(v1, v2, w, :p) == x_K2(v1, -v1 - v2 - w)
    @test y_K2(v1, v2, w, :t) == x_K2(v2+w, v1-v2)
    @test y_K3(v1, v2, w) == x_K3(v1, v2, w)
    @test y_K3(v1, v2, w, :a) == x_K3(v1, v2, w)
    @test y_K3(v1, v2, w, :p) == x_K3(v1, v2, -v1 - v2 - w)
    @test y_K3(v1, v2, w, :t) == x_K3(v2+w, v2, v1-v2)

    v1, v2, v3 = mesh[5], mesh[8], mesh[10]
    @test y_K2[v1, v3] == x_K2[v1, v3]
    @test y_K2p[v2, v3] == x_K2[v2, v3]
    @test y_K3[v1, v2, v3] == x_K3[v1, v2, v3]
end


# @testset "ReducibleVertex"
begin
    T = 0.5
    numK1 = 10
    numK2 = (5, 5)
    numK3 = (3, 3)
    γ = fdDGAsolver.ReducibleVertex(T, :a, numK1, numK2, numK3)
    @test MatsubaraFunctions.temperature(γ) == T
    @test fdDGAsolver.numK1(γ) == numK1
    @test fdDGAsolver.numK2(γ) == numK2
    @test fdDGAsolver.numK3(γ) == numK3
    @test length(γ) == (2numK1 - 1) + 2numK2[1] * (2numK2[2] - 1) + (2numK3[1])^2 * (2numK3[2] - 1)
    @test length(flatten(γ)) == length(γ)

    # Test copy
    γ.K1.data .= rand(size(γ.K1.data)...)
    γ.K2.data .= rand(size(γ.K2.data)...)
    γ.K3.data .= rand(size(γ.K3.data)...)
    γ_copy = copy(γ)
    @test γ == γ_copy

    set!(γ_copy.K1, 0)
    set!(γ_copy.K2, 0)
    set!(γ_copy.K3, 0)
    @test γ != γ_copy
    @test MatsubaraFunctions.absmax(γ) > 0
    @test MatsubaraFunctions.absmax(γ_copy) == 0

    # Test flatten and unflatten
    unflatten!(γ_copy, flatten(γ))
    @test γ == γ_copy

    # Test reduce!
    fdDGAsolver.reduce!(γ)

    # Test evaluation
    vInf = MatsubaraFrequency(T, 10^10, Fermion)
    v1 = value(meshes(γ.K3, 1)[3])
    v2 = value(meshes(γ.K3, 2)[4])
    w  = value(meshes(γ.K3, 3)[5])
    @test γ(v1, v2, w) ≈ γ.K1[w] + γ.K2[v1, w] + γ.K2[v2, w] + γ.K3[v1, v2, w]
    @test γ(vInf, vInf, w) ≈ γ.K1[w]
    @test γ(v1, vInf, w) ≈ γ.K1[w] + γ.K2[v1, w]
    @test γ(vInf, v2, w) ≈ γ.K1[w] + γ.K2[v2, w]

    # Test IO
    testfile = dirname(@__FILE__) * "/test.h5"
    file = h5open(testfile, "w")
    save!(file, "f", γ)
    close(file)

    file = h5open(testfile, "r")
    γp = fdDGAsolver.load_reducible_vertex(file, "f")
    @test γ == γp
    close(file)

    rm(testfile; force=true)
end