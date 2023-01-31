```@meta
CurrentModule = HePPCAT
```

# HePPCAT: HEteroscedastic Probabilistic PCA Technique

Documentation for [HePPCAT.jl](https://github.com/dahong67/HePPCAT.jl).

> 👋 *This package provides research code and work is ongoing.
> If you are interested in using it in your own research,
> **I'd love to hear from you and collaborate!**
> Feel free to write: [dahong67@wharton.upenn.edu](mailto:dahong67@wharton.upenn.edu)*

## What is HePPCAT?

**HePPCAT** is a probabilistic **Principal Component Analysis (PCA)** technique
**for data that has samples with heterogeneous quality**,
i.e., noise that is *[heteroscedastic](https://en.wikipedia.org/wiki/Heteroscedasticity) across samples*.

**Illustration:**
data points
with a group of *noisier samples* (blue points)
and a group of *cleaner samples* (green points).

```@eval
using HePPCAT
using CairoMakie, StableRNGs

## Setup
d = 2
n = [400,100]
v = [4,0.04]
F = ones(d,1)/sqrt(d)

## Generate data
rng = StableRNG(0)
X = [F*randn(rng,nl)' + sqrt(vl)*randn(rng,d,nl) for (nl,vl) in zip(n,v)]

## Compute estimates
models = [
    "(Homoscedastic) PPCA"           => heppcat(X,1,0),    # init is homoscedastic PPCA
    "Heteroscedastic PPCA (HePPCAT)" => heppcat(X,1,1000)
]

## Plot
datacolors = [:dodgerblue1,:green3]
fig = Figure(; resolution=(800,600))
for (idx,(title,M)) in enumerate(models)
    # Data and axes
    ax = Axis(fig[1,idx]; title=title)

    # Latent axis
    text!(ax,[(5.7,5.7)]; text="true component",
        fontsize=16.0,align=(:right,:top),rotation=pi/4)
    ablines!(ax,[0.0],[1.0]; linewidth=3,color=:black)

    # Data points
    for (Xl,color) in zip(X,datacolors)
        scatter!(ax,Xl[1,:],Xl[2,:]; strokewidth=0.2,markersize=6,color=color)
    end

    # Estimate
    text!(ax,[(M.U[1],M.U[2]).*([4.5,5.7][idx]*sqrt(2))]; text="estimate",
        fontsize=16.0,align=(:right,:bottom),rotation=atan(M.U[2]/M.U[1]),
        color=:darkorange1)
    ablines!(ax,[0.0],[M.U[2]/M.U[1]]; linewidth=3,color=:darkorange1)

    # Formatting
    ax.autolimitaspect = 1
    ax.xticks = 0:0
    ax.yticks = 0:0
    limits!(ax,(-6.25,6.25),(-6.25,6.25))
    hidedecorations!(ax,grid=false)

    # Noise variance estimates
    ax = Axis(fig[2,idx])
    scatter!(ax,reduce(vcat,fill.(M.v,n));
        color=reduce(vcat,fill.(datacolors,n)),strokecolor=:transparent,markersize=2)
    ax.xticks = [0,n[1],sum(n)]
    ax.yticks = v
    limits!(ax,(0,sum(n)),(-1,5))
    hidedecorations!(ax,grid=false)
end

# Legend
leg = Legend(fig[3,:],
    [MarkerElement(marker=:circle,color=c,strokecolor=:transparent) for c in datacolors],
    ["Noisier samples","Cleaner samples"]
)
leg.labelsize = 15
leg.tellheight = true
leg.orientation = :horizontal
leg.framevisible = false

# Format and save
Label(fig[1,1,Left()],"data and estimated components",fontsize=14,rotation=pi/2)
Label(fig[2,1,Left()],"est. noise\nvariances",fontsize=14,rotation=pi/2)
rowsize!(fig.layout, 2, Relative(1/6))

save("illustration-2D.png",fig); nothing
```

![2D illustration](illustration-2D.png)

Homoscedastic PPCA estimates *only one* noise variance for the whole data,
and treats samples as though they were all equally noisy.
Recovery of latent components can degrade a lot due to the noisier samples,
even though the rest of the samples are relatively clean.

**HePPCAT estimates latent components along with *separate* noise variances for each group.
It accounts for heterogeneous quality among the samples and is generally more robust.**

*It's not just a ["cool cat"](https://en.wiktionary.org/wiki/hepcat)!*

## How to cite

Please cite the following paper for this technique:
> David Hong, Kyle Gilman, Laura Balzano, Jeffrey A. Fessler.
> "HePPCAT: Probabilistic PCA for Data with Heteroscedastic Noise",
> IEEE Transactions on Signal Processing 69:4819-4834, Aug. 2021.
> [https://doi.org/10.1109/TSP.2021.3104979](https://doi.org/10.1109/TSP.2021.3104979)
> [https://arxiv.org/abs/2101.03468](https://arxiv.org/abs/2101.03468).

In BibTeX form:
```bibtex
@article{hgbf2021heppcat,
  title   = "{HePPCAT}: Probabilistic {PCA} for Data with Heteroscedastic Noise",
  author  = "David Hong and Kyle Gilman and Laura Balzano and Jeffrey A. Fessler",
  journal = "{IEEE} Transactions on Signal Processing",
  year    = "2021",
  volume  = "69",
  pages   = "4819--4834",
  DOI     = "10.1109/tsp.2021.3104979",
}
```
