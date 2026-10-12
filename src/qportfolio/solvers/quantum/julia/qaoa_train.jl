#!/usr/bin/env julia
"""QAOA parameter training for ParetoQ — statevector backend.

Usage:
    julia qaoa_train.jl <input.json> <output.json>

Input JSON fields
-----------------
pauli_terms_cost   list of {label, coeff_re, coeff_im}
pauli_terms_mixer  list of {label, coeff_re, coeff_im}  (kept for compatibility)
n                  number of qubits (max 12 — proxy training for larger problems)
k                  cardinality (Hamming weight of Dicke initial state)
p                  QAOA depth (layers)
n_iters            max optimizer iterations
x0                 (2p,) initial parameters [γ₀…γ_{p-1}, β₀…β_{p-1}]
seed               random seed (reserved for stochastic optimizers)
optimizer          "BFGS" or "NelderMead"
mixer_topology     "ring" or "chain" (default "ring")

Output JSON fields
------------------
params_gammas   (p,) optimized γ
params_betas    (p,) optimized β
final_energy    ⟨H_C⟩ at optimized parameters
converged       bool
n_iterations    iterations performed
history         list of {iter, energy}
runtime         training wall-clock seconds
backend         "statevector"

Pauli-string convention (matches Qiskit SparsePauliOp)
-------------------------------------------------------
A length-n string maps to a 2^n × 2^n matrix:
    matrix = kron(mat(s[1]), mat(s[2]), …, mat(s[n]))
where s[1] (1-indexed, leftmost) acts on qubit n-1 and s[n] acts on qubit 0.

Mixer convention
----------------
The mixer is applied as a sequential product of per-edge XXPlusYY operations,
matching Qiskit's XXPlusYYGate circuit exactly (same Trotter decomposition).
For each edge (q1, q2) in order:
    exp(-i · β · (X_{q1}X_{q2} + Y_{q1}Y_{q2}))
acts in the {|q1=0,q2=1⟩, |q1=1,q2=0⟩} subspace as:
    cos(2β)·|q1=0,q2=1⟩ - i·sin(2β)·|q1=1,q2=0⟩  (and conjugate)

Performance note
----------------
Eigendecompositions of H_C is computed ONCE before the optimization
loop. Each energy evaluation then only performs O(2^n) matrix-vector products
(multiply by eigenvector matrix and its conjugate transpose) rather than
O(2^{3n}) eigendecompositions.
"""

# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------
import Pkg
Pkg.add(["JSON", "Optim"]; io=devnull)

using JSON
using LinearAlgebra
using Optim

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
const _I2 = ComplexF64[1 0; 0 1]
const _X  = ComplexF64[0 1; 1 0]
const _Y  = ComplexF64[0 -im; im 0]
const _Z  = ComplexF64[1 0; 0 -1]

const _PAULI_MAT = Dict('I' => _I2, 'X' => _X, 'Y' => _Y, 'Z' => _Z)

# ===========================================================================
# STATEVECTOR BACKEND
# ===========================================================================

# Build 2^n × 2^n matrix from a single Pauli-string term.
function pauli_term_matrix(label::String, coeff::ComplexF64)::Matrix{ComplexF64}
    mat = _PAULI_MAT[label[1]]
    for i in 2:length(label)
        mat = kron(mat, _PAULI_MAT[label[i]])
    end
    return coeff .* mat
end

# Assemble Hamiltonian from list of Pauli terms.
function build_matrix(terms::Vector, n::Int)::Matrix{ComplexF64}
    dim = 1 << n
    H = zeros(ComplexF64, dim, dim)
    for t in terms
        coeff = ComplexF64(t["coeff_re"], t["coeff_im"])
        H .+= pauli_term_matrix(t["label"], coeff)
    end
    return H
end

# Dicke state |D_k^n⟩ as dense statevector.
function dicke_state_sv(n::Int, k::Int)::Vector{ComplexF64}
    dim = 1 << n
    sv = zeros(ComplexF64, dim)
    for j in 1:dim
        if count_ones(j - 1) == k
            sv[j] = 1.0
        end
    end
    sv ./= sqrt(float(binomial(n, k)))
    return sv
end

# ---------------------------------------------------------------------------
# Precomputed eigendecomposition for the cost Hamiltonian.
# ---------------------------------------------------------------------------

struct EigenFact
    values::Vector{Float64}
    vectors::Matrix{ComplexF64}
end

function precompute_eigen(H::Matrix{ComplexF64})::EigenFact
    F = eigen(Hermitian(H))
    return EigenFact(F.values, F.vectors)
end

function apply_unitary_fast(F::EigenFact, theta::Float64, psi::Vector{ComplexF64})::Vector{ComplexF64}
    # exp(-i·θ·H)|ψ⟩ = V · (exp(-i·θ·d) .* (V†·ψ))
    v = F.vectors' * psi
    v .*= exp.(-im .* theta .* F.values)
    return F.vectors * v
end

# ⟨ψ|H|ψ⟩ using precomputed H = V·D·V†
function expectation_fast(F::EigenFact, psi::Vector{ComplexF64})::Float64
    v = F.vectors' * psi
    return real(dot(F.values, abs2.(v)))
end

# ---------------------------------------------------------------------------
# Per-edge XY mixer — matches Qiskit's XXPlusYYGate sequential application.
#
# For edge (q1, q2), applies exp(-i·β·(X_{q1}X_{q2} + Y_{q1}Y_{q2})) in-place.
# Acts in the {|q1=0,q2=1⟩, |q1=1,q2=0⟩} subspace:
#   new[q1=0,q2=1] = cos(2β)·old[q1=0,q2=1] - i·sin(2β)·old[q1=1,q2=0]
#   new[q1=1,q2=0] = -i·sin(2β)·old[q1=0,q2=1] + cos(2β)·old[q1=1,q2=0]
# ---------------------------------------------------------------------------

function apply_xy_edge!(sv::Vector{ComplexF64}, beta::Float64, q1::Int, q2::Int)
    c  = cos(2.0 * beta)
    s  = -im * sin(2.0 * beta)
    b1 = 1 << q1
    b2 = 1 << q2
    for idx in 0:(length(sv) - 1)
        if (idx >> q1) & 1 == 0 && (idx >> q2) & 1 == 1
            k    = idx ⊻ b1 ⊻ b2
            jj   = idx + 1   # Julia 1-indexed
            kk   = k   + 1
            sv_j = sv[jj]
            sv_k = sv[kk]
            sv[jj] = c * sv_j + s * sv_k
            sv[kk] = s * sv_j + c * sv_k
        end
    end
end

function mixer_edges(n::Int, topology::String)::Vector{Tuple{Int,Int}}
    if topology == "ring"
        return Tuple{Int,Int}[(i, (i + 1) % n) for i in 0:(n - 1)]
    elseif topology == "chain"
        return Tuple{Int,Int}[(i, i + 1) for i in 0:(n - 2)]
    else
        error("Unsupported mixer_topology: $topology")
    end
end

function apply_xy_mixer!(sv::Vector{ComplexF64}, beta::Float64, edges::Vector{Tuple{Int,Int}})
    for (q1, q2) in edges
        apply_xy_edge!(sv, beta, q1, q2)
    end
end

# ---------------------------------------------------------------------------
# QAOA energy — cost uses eigendecomposition, mixer uses per-edge circuit.
# ---------------------------------------------------------------------------

function qaoa_energy_sv(
    params::Vector{Float64},
    F_C::EigenFact,
    edges::Vector{Tuple{Int,Int}},
    psi0::Vector{ComplexF64},
    p::Int,
)::Float64
    gammas = params[1:p]
    betas  = params[p+1:2p]
    psi = copy(psi0)
    for l in 1:p
        psi = apply_unitary_fast(F_C, gammas[l], psi)
        apply_xy_mixer!(psi, betas[l], edges)
    end
    return expectation_fast(F_C, psi)
end

# ===========================================================================
# Main
# ===========================================================================
function main(input_path::String, output_path::String)
    t_start = time()

    input    = JSON.parsefile(input_path)
    n        = Int(input["n"])
    k        = Int(input["k"])
    p        = Int(input["p"])
    niters   = Int(input["n_iters"])
    x0       = Float64.(input["x0"])
    opt_name = input["optimizer"]
    topology = get(input, "mixer_topology", "ring")

    if n > 12
        error("Statevector training is limited to n ≤ 12 qubits (got n=$n). " *
              "Dense matrices are 2^n × 2^n: n=12 → 256 MB, n=15 → 17 GB. " *
              "QAOASolver should use a proxy Hamiltonian for larger problems.")
    end

    println(stderr, "[Julia QAOA] n=$n, k=$k, p=$p, optimizer=$opt_name, n_iters=$niters")
    flush(stderr)

    H_C  = build_matrix(input["pauli_terms_cost"], n)
    psi0 = dicke_state_sv(n, k)
    edges = mixer_edges(n, topology)

    # Precompute cost eigendecomposition once.
    println(stderr, "[Julia QAOA] Precomputing eigendecompositions (dim=$(1<<n))...")
    flush(stderr)
    F_C = precompute_eigen(H_C)
    println(stderr, "[Julia QAOA] Eigendecompositions done, starting optimization...")
    flush(stderr)

    call_count = Ref(0)
    history    = Vector{Dict{String,Any}}()

    function neg_energy_sv(params::Vector{Float64})::Float64
        call_count[] += 1
        ev = qaoa_energy_sv(params, F_C, edges, psi0, p)
        push!(history, Dict{String,Any}("iter" => call_count[], "energy" => ev))
        return -ev
    end

    opts = Optim.Options(iterations=niters, store_trace=false, show_trace=false)
    opt_result = if opt_name == "BFGS"
        Optim.optimize(neg_energy_sv, x0, BFGS(), opts)
    elseif opt_name == "NelderMead"
        Optim.optimize(neg_energy_sv, x0, NelderMead(), opts)
    else
        error("Unknown optimizer: $opt_name")
    end

    best     = Optim.minimizer(opt_result)
    final_ev = -Optim.minimum(opt_result)

    println(stderr, "[Julia QAOA] Done: $(call_count[]) energy calls, " *
            "converged=$(Optim.converged(opt_result)), " *
            "runtime=$(round(time()-t_start, digits=2))s")
    flush(stderr)

    output = Dict(
        "params_gammas" => best[1:p],
        "params_betas"  => best[p+1:2p],
        "final_energy"  => final_ev,
        "converged"     => Optim.converged(opt_result),
        "n_iterations"  => Optim.iterations(opt_result),
        "history"       => history,
        "runtime"       => time() - t_start,
        "backend"       => "statevector",
    )

    open(output_path, "w") do io
        JSON.print(io, output)
    end
end

if length(ARGS) != 2
    println(stderr, "Usage: julia qaoa_train.jl <input.json> <output.json>")
    exit(1)
end

main(ARGS[1], ARGS[2])
