```@meta
CurrentModule = HePPCAT
```

# Quick Start: How to use HePPCAT

## Step 0: Installation
This package is registered and can be installed
via the [package manager](https://docs.julialang.org/en/v1/stdlib/Pkg/).
```julia-repl
pkg> add HePPCAT  # type `]` to access this `pkg>` prompt
```

## Step 1: Form list of data matrices
HePPCAT expects the samples to be arranged as columns in matrices,
where each matrix corresponds to one noise variance group:
```math
\mathbf{Y}_1 =
\underbrace{
\begin{bmatrix}
    | & & | \\
    \mathbf{y}_{1,1} & \cdots & \mathbf{y}_{1,n_1} \\
    | & & |
\end{bmatrix}
}_{d \times n_1 \text{ matrix}}
\quad \cdots \quad
\mathbf{Y}_L =
\underbrace{
\begin{bmatrix}
    | & & | \\
    \mathbf{y}_{L,1} & \cdots & \mathbf{y}_{L,n_L} \\
    | & & |
\end{bmatrix}
}_{d \times n_L \text{matrix}}
```
i.e.,
the samples ``\mathbf{y}_{1,1},\dots,\mathbf{y}_{1,n_1}``
will have one associated noise variance estimate,
the samples ``\mathbf{y}_{2,1},\dots,\mathbf{y}_{2,n_2}``
will have one associated noise variance estimate,
and so on.

The following code generates a synthetic set of data
(for illustration):
```@example tutorial
d = 100        # number of features (ambient dimension)
k = 1          # number of factors (latent dimension)

n = [800,200]  # number of samples in each group/block
v = [10,0.01]  # noise variance for each group/block
L = length(n)  # number of groups/blocks

F = randn(d,k)/sqrt(d) # synthetic factor matrix
Y = [F*randn(k,n[l]) + sqrt(v[l])*randn(d,n[l]) for l in 1:L]
```
Namely, `Y` is a list (a `Vector`) of matrices
containing `d = 100` dimensional samples:
+ `Y[1]` has `n[1] = 800` samples with a noise variance of `v[1] = 10`
+ `Y[2]` has `n[2] = 200` samples with a noise variance of `v[2] = 0.01`.

## Step 2: Run the main function `heppcat`
With the package installed and the data `Y` prepared,
simply load the package and run `heppcat`:
```@repl tutorial
using HePPCAT
model = heppcat(Y,k,100)  # run 100 iterations
```

The output is a `HePPCATModel`
that contains the factor matrix and noise variance estimates.
Extracting them is as simple as running:
```@repl tutorial
model.F
model.v
```

**That's it!**

For more examples, see the [paper code repo](https://gitlab.com/dahong/heteroscedastic-probabilistic-pca)!
