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
scene, layout = layoutscene(20,resolution=(800,600))
datacolors = [:dodgerblue1,:green3]
for (idx,(title,M)) in enumerate(models)
    # Data and axes
    ax = layout[1,idx] = LAxis(scene,title=title)

    # Latent axis
    text!(ax,"true component",position=(5.7,5.7),textsize=0.55,align=(:right,:top),rotation=pi/4)
    lines!(ax,-6.25..6.25,identity,linewidth=3,color=:black)

    # Data points
    for (Xl,color) in zip(X,datacolors)
        scatter!(ax,Xl[1,:],Xl[2,:],strokewidth=0.2,markersize=6,color=color)
    end

    # Estimate
    text!(ax,"estimate",position=(M.U[1],M.U[2]).*([4.5,5.7][idx]*sqrt(2)),textsize=0.55,align=(:right,:bottom),rotation=atan(M.U[2]/M.U[1]),color=:darkorange1)
    lines!(ax,-6.25..6.25,x->x*M.U[2]/M.U[1],linewidth=3,color=:darkorange1)

    # Formatting
    ax.autolimitaspect = 1
    ax.xticks = 0:0
    ax.yticks = 0:0
    limits!(ax,(-6.25,6.25),(-6.25,6.25))
    hidedecorations!(ax,grid=false)

    # Noise variance estimates
    ax = layout[2,idx] = LAxis(scene)
    scatter!(ax,reduce(vcat,fill.(M.v,n)),color=reduce(vcat,fill.(datacolors,n)),strokecolor=:transparent,markersize=2)
    # scatter!(ax,fill(-1,sum(n)),color=reduce(vcat,fill.(datacolors,n)),strokecolor=:transparent,markersize=5)
    # scatter!(ax,reduce(vcat,fill.(v,n)),color=:black,strokecolor=:transparent,markersize=5)
    # scatter!(ax,reduce(vcat,fill.(M.v,n)),color=:darkorange1,strokecolor=:transparent,markersize=3)
    ax.xticks = [0,n[1],sum(n)]
    ax.yticks = v
    limits!(ax,(0,sum(n)),(-1,5))
    hidedecorations!(ax,grid=false)
end

# Legend
leg = layout[end+1,:] = LLegend(scene,
    [MarkerElement(marker=:circle,color=c,strokecolor=:transparent) for c in datacolors],
    ["Noisier samples","Cleaner samples"]
)
leg.labelsize = 15
leg.tellheight = true
leg.orientation = :horizontal
leg.framevisible = false

# Format and save
layout[1,1,Left()] = LText(scene,"data and estimated components",textsize=14,rotation=pi/2)
layout[2,1,Left()] = LText(scene,"estimated\nnoise vars.",textsize=14,rotation=pi/2)
rowsize!(layout, 2, Relative(1/6))

CairoMakie.save("illustration-2D.png",scene)
scene
