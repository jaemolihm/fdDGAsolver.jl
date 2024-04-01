"""
# Spin channel
Under SU(2) symmetry, there are six nonzero components to the vertex, which are related as
`Γ↑↑↑↑ = Γ↓↓↓↓`, `Γ↑↑↓↓ = Γ↓↓↑↑`, `Γ↑↓↓↑ = Γ↓↑↑↓ = Γ↑↑↑↑ - Γ↑↑↓↓`.

We use different spin parametrizations for different channels:
- a: density `:Da = Γ↑↑↑↑ + Γ↑↑↓↓` and magnetic `:Ma = Γ↑↑↑↑ - Γ↑↑↓↓`.
- p: singlet `:Sp = Γ↑↓↓↑ - Γ↑↑↓↓` and triplet  `:Tp = Γ↑↑↓↓ + Γ↑↓↓↑`.
- t: density `:Dt = Γ↑↑↑↑ + Γ↑↓↓↑` and magnetic `:Mt = Γ↑↑↑↑ - Γ↑↓↓↑`.

Conversion between the two spin parametrizations are done using the
`su2_convert_spin_channel` function.
"""

"""
    convert_spin_channel(C_from :: Symbol, C_to :: Symbol, x1, x2)
Convert between the natural spin parametrization of each channel.
"""
function convert_spin_channel(C_from :: Symbol, C_to :: Symbol, x1, x2)
    C_from === C_to && return (x1, x2)
    (C_from, C_to) === (:a, :p) && return (x1 * -1/2 + x2 *  3/2, x1 *  1/2 + x2 *  1/2)
    (C_from, C_to) === (:p, :a) && return (x1 * -1/2 + x2 *  3/2, x1 *  1/2 + x2 *  1/2)
    (C_from, C_to) === (:t, :a) && return (x1 *  1/2 + x2 *  3/2, x1 *  1/2 + x2 * -1/2)
    (C_from, C_to) === (:a, :t) && return (x1 *  1/2 + x2 *  3/2, x1 *  1/2 + x2 * -1/2)
    (C_from, C_to) === (:t, :p) && return (x1 *  1/2 + x2 * -3/2, x1 *  1/2 + x2 *  1/2)
    (C_from, C_to) === (:p, :t) && return (x1 *  1/2 + x2 *  3/2, x1 * -1/2 + x2 *  1/2)
    throw(ArgumentError("Wrong channels $C_to and $C_from"))
end

"""
    su2_bare_vertex(C :: Symbol, U)
Multiply coefficients for the bare vertices for the two SU(2) spin channels in channel `C`.
"""
function su2_bare_vertex(C :: Symbol, U)
    C === :a && return (U, -U)
    C === :p && return (-2 * U, 0 * U)
    C === :t && return (-U, U)
    throw(ArgumentError("Wrong channel $C"))
end

export
    convert_spin_channel,
    su2_bare_vertex