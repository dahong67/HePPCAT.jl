## Illustration in 2D
# note: behavior is generally less consistent in lower dimensions but 2D is convenient for illustration/visualization

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
# X = reshape.(collect.(eachcol(reduce(hcat,X))),:,1) # flatten to give each sample its own noise variance estimate

## Compute estimates
models = [
    "(Homoscedastic) PPCA"           => heppcat(X,1,0),    # init is homoscedastic PPCA
    "Heteroscedastic PPCA (HePPCAT)" => heppcat(X,1,1000)
]

## Plot
scene, layout = layoutscene(20,resolution=(800,500))
datacolors = [:dodgerblue1,:green3]
for (idx,(title,M)) in enumerate(models)
    # Data and axes
    ax = layout[1,idx] = LAxis(scene,title=title)

    # Latent axis
    text!(ax,"latent/true axis",position=(5.6,5.6),textsize=0.55,align=(:right,:top),rotation=pi/4)
    lines!(ax,-6.25..6.25,identity,linewidth=3,color=:black)

    # Data points
    for (Xl,color) in zip(X,datacolors)
        scatter!(ax,Xl[1,:],Xl[2,:],strokewidth=0.2,markersize=6,color=color)
    end

    # Estimate
    text!(ax,"estimate",position=(M.U[1]/M.U[2],1).*(idx == 1 ? 5.8 : 5.6),textsize=0.55,align=(:right,:bottom),rotation=atan(M.U[2]/M.U[1]),color=:darkorange1)
    lines!(ax,-6.25..6.25,x->x*M.U[2]/M.U[1],linewidth=3,color=:darkorange1)

    # Formatting
    ax.autolimitaspect = 1
    ax.xticks = 0:0
    ax.yticks = 0:0
    limits!(ax,(-6.25,6.25),(-6.25,6.25))
    hidedecorations!(ax,grid=false)

    # # Noise variance estimates
    # ax = layout[2,idx] = LAxis(scene)
    # scatter!(ax,M.v,color=[fill(datacolors[1],n[1]); fill(datacolors[2],n[2])],strokecolor=:transparent,markersize=2)
    # ax.xticks = [0,n[1],sum(n)]
    # limits!(ax,(0,sum(n)),(0,10))
    # hideydecorations!(ax,grid=false)
end

# Legend
leg = layout[end+1,:] = LLegend(scene,
    [MarkerElement(marker=:circle,color=c,strokecolor=:transparent) for c in datacolors],
    ["Noisier data","Cleaner data"]
)
leg.labelsize = 15
leg.tellheight = true
leg.orientation = :horizontal
leg.framevisible = false

# Save
CairoMakie.save("illustration-2D.png",scene)
