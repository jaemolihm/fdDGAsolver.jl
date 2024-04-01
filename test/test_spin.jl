using fdDGAsolver
using Test

@testset "spin channel" begin
    x1 = rand()
    x2 = rand()
    @test all(convert_spin_channel(:a, :a, x1, x2) .≈ (x1, x2))
    @test all(convert_spin_channel(:a, :p, x1, x2) .≈ ((-x1 + 3 * x2) / 2, ( x1 + x2) / 2))
    @test all(convert_spin_channel(:a, :t, x1, x2) .≈ (( x1 + 3 * x2) / 2, ( x1 - x2) / 2))
    @test all(convert_spin_channel(:p, :a, x1, x2) .≈ ((-x1 + 3 * x2) / 2, ( x1 + x2) / 2))
    @test all(convert_spin_channel(:p, :p, x1, x2) .≈ (x1, x2))
    @test all(convert_spin_channel(:p, :t, x1, x2) .≈ (( x1 + 3 * x2) / 2, (-x1 + x2) / 2))
    @test all(convert_spin_channel(:t, :a, x1, x2) .≈ (( x1 + 3 * x2) / 2, ( x1 - x2) / 2))
    @test all(convert_spin_channel(:t, :p, x1, x2) .≈ (( x1 - 3 * x2) / 2, ( x1 + x2) / 2))
    @test all(convert_spin_channel(:t, :t, x1, x2) .≈ (x1, x2))

    # Check if c1 -> c2 -> c1 transforms back to itself
    for c1 in (:a, :p, :t), c2 in (:a, :p, :t)
        @test all(convert_spin_channel(c2, c1, convert_spin_channel(c1, c2, x1, x2)...) .≈ (x1, x2))
    end

    # Test su2_bare_vertex
    U = 5.0
    Ua = su2_bare_vertex(:a, U)
    Up = su2_bare_vertex(:p, U)
    Ut = su2_bare_vertex(:t, U)
    @test all(Ua .≈ (U, -U))
    @test all(Up .≈ (-2U, 0))
    @test all(Ut .≈ (-U, U))
    @test all(Ua .≈ convert_spin_channel(:p, :a, Up...))
    @test all(Ua .≈ convert_spin_channel(:t, :a, Ut...))
    @test all(Up .≈ convert_spin_channel(:a, :p, Ua...))
    @test all(Up .≈ convert_spin_channel(:t, :p, Ut...))
    @test all(Ut .≈ convert_spin_channel(:a, :t, Ua...))
    @test all(Ut .≈ convert_spin_channel(:p, :t, Up...))
end
