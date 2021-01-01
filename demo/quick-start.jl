### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ ce74e976-4be2-11eb-29e9-4f2e622164a7
using HePPCAT

# ╔═╡ 21e07958-4be4-11eb-3f15-791d9e54ed12
using LinearAlgebra

# ╔═╡ 16f41744-4be3-11eb-1b34-4d680b206e76
md"""
# Quick start notebook for [`HePPCAT.jl`](https://github.com/dahong67/HePPCAT.jl)
"""

# ╔═╡ 261c7946-4be3-11eb-22a1-2fcb36cd9de6
md"""
**Step 1:** load the package.
"""

# ╔═╡ 13a9c3ae-4be3-11eb-0400-d94006c590e7
md"""
**Step 2:** form data into a `Vector` with `Matrix` elements -
each matrix is a group of samples (columns) having equal noise variance.
"""

# ╔═╡ 494732f8-4be3-11eb-22e3-9b1ac3913f41
begin # Example: generating synthetic data with heteroscedastic noise
   d = 100        # number of features (ambient dimension)
   k = 1          # number of factors (latent dimension)

   n = [800,200]  # number of samples in each group/block
   v = [10,0.01]  # noise variance for each group/block
   L = length(n)  # number of groups/blocks

   F = randn(d,k)/sqrt(d) # synthetic factor matrix
   X = [F*randn(k,n[l]) + sqrt(v[l])*randn(d,n[l]) for l in 1:L]  # synthetic data
end

# ╔═╡ 4cc04d34-4be3-11eb-31d5-0fb11143809b
md"""
This code generates an example synthetic dataset with heteroscedastic noise:
the `n[1]` = $(n[1]) samples in `X[1]` have noise variance `v[1]` = $(v[1]),
and the `n[2]` = $(n[2]) samples in `X[2]` have noise variance `v[2]` = $(v[2]).
"""

# ╔═╡ 707ac7cc-4be3-11eb-3091-4b5397319891
md"""
**Step 3:** run the `heppcat` method.
"""

# ╔═╡ a223fd5c-4be3-11eb-097b-4db63a3a9aa3
model = heppcat(X,k,100)  # run 100 iterations

# ╔═╡ ad0256d8-4be3-11eb-084b-2ffb4fece64d
md"""
For more info, use `?heppcat` to access the [docstring](https://docs.julialang.org/en/v1/manual/documentation/#Accessing-Documentation).
"""

# ╔═╡ c7db5bf8-4be3-11eb-3460-15b039b47f07
md"""
**Step 4:** extract factor matrix and noise variance estimates.
"""

# ╔═╡ e6a22602-4be3-11eb-0532-276b6a63a00b
model.F

# ╔═╡ e73c22c0-4be3-11eb-276a-afb37f0146ea
model.v

# ╔═╡ e9a19a04-4be3-11eb-0f4e-a72c20cf75e4
md"""
For more info about the returned datatype use `?HePPCATModel`
for the [docstring](https://docs.julialang.org/en/v1/manual/documentation/#Accessing-Documentation).
"""

# ╔═╡ 1fcfcefc-4be4-11eb-0041-cf96d0098e53
md"""
**Evaluating recovery:**
check recovery of the factor covariance and noise variances.
"""

# ╔═╡ 2841c4aa-4be4-11eb-206e-df2fd2696ac6
norm(model.F*model.F' - F*F')/norm(F*F')  # normalized estimation error

# ╔═╡ 2d2f42bc-4be4-11eb-043d-2723dad42b10
[model.v v]

# ╔═╡ 393e4b20-4be4-11eb-3d69-6fa84bae8d49
md"""
Compared with homoscedastic PPCA:
"""

# ╔═╡ 5b273d08-4be4-11eb-1d7b-7373113b2ca6
ppca = heppcat(X,k,0)  # initialization is homoscedastic PPCA

# ╔═╡ 5f50f2c0-4be4-11eb-1939-518d61b5fa73
norm(ppca.F*ppca.F' - F*F')/norm(F*F')  # normalized estimation error

# ╔═╡ 6281428a-4be4-11eb-0b2e-e7f9423afdb4
[ppca.v v]

# ╔═╡ 65b6c7c4-4be4-11eb-0950-b71e94e85221
md"""
`HePPCAT` accounts for heterogeneous quality among the samples
and is generally more robust.
"""

# ╔═╡ Cell order:
# ╟─16f41744-4be3-11eb-1b34-4d680b206e76
# ╟─261c7946-4be3-11eb-22a1-2fcb36cd9de6
# ╠═ce74e976-4be2-11eb-29e9-4f2e622164a7
# ╟─13a9c3ae-4be3-11eb-0400-d94006c590e7
# ╠═494732f8-4be3-11eb-22e3-9b1ac3913f41
# ╟─4cc04d34-4be3-11eb-31d5-0fb11143809b
# ╟─707ac7cc-4be3-11eb-3091-4b5397319891
# ╠═a223fd5c-4be3-11eb-097b-4db63a3a9aa3
# ╟─ad0256d8-4be3-11eb-084b-2ffb4fece64d
# ╟─c7db5bf8-4be3-11eb-3460-15b039b47f07
# ╠═e6a22602-4be3-11eb-0532-276b6a63a00b
# ╠═e73c22c0-4be3-11eb-276a-afb37f0146ea
# ╟─e9a19a04-4be3-11eb-0f4e-a72c20cf75e4
# ╟─1fcfcefc-4be4-11eb-0041-cf96d0098e53
# ╠═21e07958-4be4-11eb-3f15-791d9e54ed12
# ╠═2841c4aa-4be4-11eb-206e-df2fd2696ac6
# ╠═2d2f42bc-4be4-11eb-043d-2723dad42b10
# ╟─393e4b20-4be4-11eb-3d69-6fa84bae8d49
# ╠═5b273d08-4be4-11eb-1d7b-7373113b2ca6
# ╠═5f50f2c0-4be4-11eb-1939-518d61b5fa73
# ╠═6281428a-4be4-11eb-0b2e-e7f9423afdb4
# ╟─65b6c7c4-4be4-11eb-0950-b71e94e85221
