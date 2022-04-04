using LinearAlgebra
using DelimitedFiles
using Printf

include("barrier.jl")
include("primal_dual.jl")


experiments = readdlm("data/experiments.txt", '\t', Int, '\n')

print(experiments)

ns = experiments[1,:]

ms = experiments[2,:]

for (n, m) in zip(ns, ms)

    e = ones((n,1))
    iden = ones((n,n))

    c = randn((n,1))
    b = randn((m,1))

    A = randn((m,n))


    x = randn((n,1))
    y = randn((m,1))
    s = randn((n,1))

    # print(x)

    σ = 0.5

    μ = 1/n * dot(x, s)

    α = 0.9995

    max_iter = 500

    ε = 1e-8

    𝑶 = zeros((m,m))

    𝒐 = zeros((m,1))

    𝑶2 = zeros((m,n))

    𝑶3 = zeros((n,n))

    primal_dual(n, m, e, iden, c, b, A, x, y, s, σ, μ, α, max_iter, ε, 𝑶, 𝑶2, 𝑶3)

    # barrier(n, m, e, c, b, A, x, σ, μ, α, max_iter, ε, 𝑶, 𝒐)

end