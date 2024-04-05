function MatsubaraFunctions.add!(f1 :: MeshFunction, f2 :: MeshFunction, α :: Number) :: Nothing
    MatsubaraFunctions.debug_f1_f2(f1, f2)
    f1.data .+= f2.data .* α
    return nothing
end
