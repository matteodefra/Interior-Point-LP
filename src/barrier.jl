using LinearAlgebra
using Printf


function barrier(n, m, e, c, b, A, x, σ, μ, α, max_iter, ε, 𝑶, 𝒐)

    print("Newton Method with Logarithmic Barrier method")

    print("n = $(n), m = $(m)")

    print("iter\t\ttime\t\tf(x)-ϕ(y)\t\t∥Ax-b∥\t\t∥Ay+s-c∥\t\t∥g∥\n\n")

    for iteration=1:1:max_iter

        start = time()

        X = diagm(vec(x.^(-1)))

        # Build matrix to find the Newton directions
        𝑮 = [μ*X^2 -A' ; A 𝑶]

        𝒚 = [μ * X * e - c; 𝒐]
        
        Δ = 𝑮 \ 𝒚

        Δx, Δy = Δ[1:n], Δ[n+1:n+m]

        # Update variables
        x = x + α*Δx

        y = Δy

        s = c - A' * y

        # Compute gaps
        gap = c' * x - b' * y 

        least_squares = norm(A * x - b)

        s_norm = norm(A' * y + s - c)

        finish = time()

        time_step = finish - start

        @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" iteration time_step gap[1] least_squares s_norm norm(Δ)

        if max(gap[1], s_norm, least_squares) ≤ ε
            print("Stopping conditions satisfied!")
        end

        # Update barrier
        μ = σ * μ

    end

    print("\n\n")

end