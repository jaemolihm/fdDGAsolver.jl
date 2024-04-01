using fdDGAsolver
using MatsubaraFunctions
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