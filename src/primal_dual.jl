using LinearAlgebra



function primal_dual(n, m, e, iden, c, b, A, x, y, s, σ, μ, α, max_iter, ϵ, 𝑶, 𝑶2, 𝑶3)

    for i=1:1:max_iter

        X = diagm(vec(x))

        S = diagm(vec(s))

        # Build matrix to find the Newton directions
        𝑮 = [A 𝑶 𝑶2; 𝑶3 A' iden; S 𝑶2' X]

        𝒚 = [b - A*x; c - A' * y - s; μ * e - X*S*e]

        Δ = 𝑮 \ 𝒚

        print(size(Δ))

        Δx, Δy, Δs = Δ[1:n], Δ[n+1:n+m], Δ[n+m+1:2*n+m]

        # Update variables
        x = x + α*Δx

        s = s + α*Δs

        y = y + α*Δy

        # Compute gaps
        gap1 = c' * x - b' * y 

        gap2 = x' * s 

        display(gap2)


        # Update barrier
        μ = σ * μ

    end

end
