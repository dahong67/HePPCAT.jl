using HePPCAT
using Documenter

DocMeta.setdocmeta!(HePPCAT, :DocTestSetup, :(using HePPCAT); recursive=true)

makedocs(;
    modules=[HePPCAT],
    authors="David Hong <dahong67@wharton.upenn.edu> and contributors",
    repo="https://github.com/dahong67/HePPCAT.jl/blob/{commit}{path}#{line}",
    sitename="HePPCAT.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dahong67.github.io/HePPCAT.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dahong67/HePPCAT.jl",
    devbranch="master",
)
