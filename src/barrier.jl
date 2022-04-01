using LinearAlgebra


function barrier(n, m, e, iden, c, b, A, x, y, s, σ, μ, α, max_iter, ϵ, 𝑶, 𝒐)

    for i=1:1:max_iter

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
        gap1 = c' * x - b' * y 

        gap2 = x' * s 

        display(gap2)


        # Update barrier
        μ = σ * μ

    end

end