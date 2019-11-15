#=
Created on Friday 15 November 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Compute eigenvectors from eigenvals
https://arxiv.org/abs/1908.03795
=#

using LinearAlgebra, Test

function eigvals2eigvecs(A::Hermitian)
    n = size(A, 1)
    # eigenvalues of A
    λ = eigvals(A)
    # eigenvalues of principal submatrices
    Λ = [eigvals(A[1:n.!=i,1:n.!=i]) for i in 1:n]
    # squared eigenvalues using strange idenity
    V² = [prod((λ[i] - Λ[j][k] for k in 1:n-1)) /
                    prod((λ[i] - λ[k] for k in 1:n if k!=i))
                    for j in 1:n, i in 1:n]
    return V²
end

A = rand(20, 20) |> Hermitian;
@test eigvecs(A).^2 ≈ eigvals2eigvecs(A)  # pass
