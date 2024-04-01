using fdDGAsolver
using MatsubaraFunctions
using Test

@testset begin
    T = 1.0
    N = 10
    mesh = MatsubaraMesh(T, N, Fermion)
    x = MeshFunction((mesh,), rand(length(mesh)))
    y = SU2Vertex_K1(x, :a, :Da, 3.)
    @test y[mesh[1]] === x[mesh[1]]

    v1, v2, w = -10., 9., 5.
    @test y(v1, v2, w) == x(w)
    @test y(v1, v2, w, :a) == x(w)
    @test y(v1, v2, w, :p) == x(-v1 - v2 - w)
    @test y(v1, v2, w, :t) == x(v1 - v2)
    @test y(v1, v2, 10_000.) == 3.
end
