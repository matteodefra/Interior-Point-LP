using LinearAlgebra
using Printf


function primal_dual(n, m, e, iden, c, b, A, x, y, s, r, μ, α, max_iter, ε, 𝑶, 𝑶2, 𝑶3)

    print("Primal-Dual Interior Point method\n")

    print("n = $(n), m = $(m)\n")

    print("iter\t\ttime\t\ts⋅x\t\t∥Ax-b∥\t\t∥Ay+s-c∥\t\t∥g∥\n\n")

    for iteration=1:1:max_iter

        start = time()

        X = diagm(vec(x))

        S = diagm(vec(s))

        # Shrink μ
        μ = α * (dot(x, s) / iteration)

        # Build matrix to find the Newton directions
        𝑮 = [A 𝑶 𝑶2; 𝑶3 A' iden; S 𝑶2' X]

        𝒚 = [b - A*x; c - A' * y - s; μ * e - X*S*e]

        Δ = 𝑮 \ 𝒚

        Δx, Δy, Δs = Δ[1:n], Δ[n+1:n+m], Δ[n+m+1:2*n+m]

        # Update variables
        x = x + α*Δx

        s = s + α*Δs

        y = y + α*Δy

        # Compute gaps
        least_squares = norm(A * x - b)

        s_norm = norm(A' * y + s - c)

        complementary_slackness = (s' * x)[1]

        g_norm = norm(Δ)

        finish = time()

        time_step = finish - start

        @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" iteration time_step complementary_slackness least_squares s_norm g_norm

        # Check stopping condition
        if max(complementary_slackness, s_norm, least_squares) ≤ ε
            print("Stopping conditions satisfied!")
            break
        end

    end

    print("\n\n")

end
