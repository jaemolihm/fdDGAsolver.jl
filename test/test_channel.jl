using fdDGAsolver
using Test

@testset "frequency channel" begin
    using fdDGAsolver: shuffle_to_standard, shuffle_to_channel,
                       frequency_to_standard, frequency_to_channel
    for C in (:a, :p, :t)
        inds_standard = shuffle_to_standard(C, (1, 2, 3, 4))
        @test shuffle_to_channel(C, inds_standard) == (1, 2, 3, 4)

        vvw = rand(3)
        v1234 = frequency_to_standard(C, vvw...)
        @test all(frequency_to_channel(C, v1234...) .≈ vvw)
    end

    for C in (:a, :p, :t)
        for v1 in -2:2, v2 in -2:2, w in -2:2
            x = frequency_to_standard(C, v1, v2, w)
            y = frequency_to_channel(C, x...)
            @test sum(x) == 0
            @test y == (v1, v2, w)
        end
    end

    # # Check vertex and bubble parametrizations are consistent
    # for C in (:A, :P, :T)
    #     # Vertex indices 3 and 4 belong to the left vertex, and 1 and 2 to the right.
    #     inds_vertex = ManyBodyVertex.shuffle_to_standard(C, (1, 2, 3, 4))
    #     inds_bubble = ManyBodyVertex._bubble_indices(C, (3, 4, 1, 2))

    #     # First and third indices of inds_bubble are incoming, second and fourth are
    #     # outgoing in the vertex standard representation.
    #     @test iseven(inds_vertex[inds_bubble[1]])
    #     @test isodd(inds_vertex[inds_bubble[2]])
    #     @test iseven(inds_vertex[inds_bubble[3]])
    #     @test isodd(inds_vertex[inds_bubble[4]])

    #     # Frequencies of the vertex. Multiply -1 for incoming legs.
    #     for f in (:MF, :ZF, :KF)
    #         for (vL, vB, vR, w) in ((1, 3, -6, 23), (0, -6, 4, 22))
    #             v1234_L = ManyBodyVertex.frequency_to_standard(Val(f), C, vL, vB, w) .* (1, -1, 1, -1)
    #             v1234_R = ManyBodyVertex.frequency_to_standard(Val(f), C, vB, vR, w) .* (1, -1, 1, -1)
    #             if f === :MF
    #                 # For imaginary frequencies, -v requires additional subtraction of 1.
    #                 v1234_L = v1234_L .- (0, 1, 0, 1)
    #                 v1234_R = v1234_R .- (0, 1, 0, 1)
    #             end
    #             v1, v2 = ManyBodyVertex._bubble_frequencies(Val(f), C, vB, w)

    #             # inds_bubble[2] -- v1 --> inds_bubble[1]
    #             for i in 1:2
    #                 v1234_vertex = inds_bubble[i] <= 2 ? v1234_R : v1234_L
    #                 @test v1 == v1234_vertex[inds_vertex[inds_bubble[i]]]
    #             end

    #             # inds_bubble[4] -- v2 --> inds_bubble[3]
    #             for i in 3:4
    #                 v1234_vertex = inds_bubble[i] <= 2 ? v1234_R : v1234_L
    #                 @test v2 == v1234_vertex[inds_vertex[inds_bubble[i]]]
    #             end
    #         end
    #     end
    # end
end
