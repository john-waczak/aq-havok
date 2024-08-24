montserrat_path = "./font-files/montserrat"
@assert ispath(montserrat_path)

mints_colors = [
    colorant"#3cd184", # mint green
    colorant"#f97171", # dark coral
    colorant"#1e81b0", # dark blue
    colorant"#66beb2", # dark blue-green
    colorant"#f99192", # light coral
    colorant"#8ad6cc", # middle blue-green
    colorant"#3d6647", # dark green
#    colorant"#8FDDDF", # middle blue
]


mints_font = (;
              regular = joinpath(montserrat_path, "static", "Montserrat-Regular.ttf"),
              italic = joinpath(montserrat_path, "static", "Montserrat-Italic.ttf"),
              bold = joinpath(montserrat_path, "static", "Montserrat-Bold.ttf"),
              bold_italic = joinpath(montserrat_path, "static", "Montserrat-BoldItalic.ttf"),
              )



# note: to get a specific attribute (for example, Axis attribute x), go into help mode with ?
# and type Axis.x
# for a full list, type ? then Axis.

mints_theme = Theme(
    fontsize=17,
    fonts = mints_font,
    colormap = :viridis,
    palette = (
        color = mints_colors,
        patchcolor = mints_colors,
    ),
    cycle = [[:linecolor, :markercolor,] => :color,],
    Axis=(
        xlabelsize=15,                   ylabelsize=15,
        xticklabelsize=13,               yticklabelsize=13,
        xticksize=3,                     yticksize=3,
        xminorgridvisible=true,          yminorgridvisible=true,
        xgridwidth=2,                    ygridwidth=2,
        xminorgridwidth=1,               yminorgridwidth=1,
        xminorticks=IntervalsBetween(5), yminorticks=IntervalsBetween(5)
    ),
    Colorbar=(
        fontsize=13,
    )
)




using Statistics, StatsBase
r_squared(ŷ, y) =  1 - sum(abs2.(ŷ .- y))/sum(abs2.(mean(y) .- y))

function scatter_results(
    y,
    ŷ,
    ytest,
    ŷtest,
    varname
    )

    fig = Figure();
    ga = fig[1, 1] = GridLayout()
    axtop = Axis(ga[1, 1];
                leftspinevisible = false,
                rightspinevisible = false,
                bottomspinevisible = false,
                topspinevisible = false,
                )
    axmain = Axis(ga[2, 1], xlabel = "True $(varname)", ylabel = "Predicted $(varname)")
    axright = Axis(ga[2, 2];
                  leftspinevisible = false,
                  rightspinevisible = false,
                  bottomspinevisible = false,
                  topspinevisible = false,
                  )

    linkyaxes!(axmain, axright)
    linkxaxes!(axmain, axtop)

    minval, maxval = extrema([extrema(y)..., extrema(ytest)..., extrema(ŷ)..., extrema(ŷtest)...])
    δ_edge = 0.1*(maxval-minval)

    l1 = lines!(axmain, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)
    s1 = scatter!(axmain, y, ŷ, alpha=0.75)
    s2 = scatter!(axmain, ytest, ŷtest, marker=:rect, alpha=0.75)

    labels=[
        "Training R²=$(round(r_squared(ŷ, y), digits=3)) (n=$(length(y)))",
        "Testing   R²=$(round(r_squared(ŷtest, ytest), digits=3)) (n=$(length(ytest)))",
        "1:1"
    ]

    # leg = Legend(ga[1, 2], [s1, s2, l1], labels)
    leg = axislegend(axmain, [s1, s2, l1], labels; position=:lt, labelsize=13)

    density!(axtop, y, color=(mints_colors[1], 0.5), strokecolor=mints_colors[1], strokewidth=2)
    density!(axtop, ytest, color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=2)

    density!(axright, ŷ, direction = :y, color=(mints_colors[1], 0.5), strokecolor=mints_colors[1], strokewidth=2)
    density!(axright, ŷtest, direction = :y, color=(mints_colors[2], 0.5), strokecolor=mints_colors[2], strokewidth=2)

    hidedecorations!(axtop)
    hidedecorations!(axright)
    #leg.tellheight = true
    rowsize!(ga, 1, Relative(0.1))
    colsize!(ga, 2, Relative(0.1))

    colgap!(ga, 0)
    rowgap!(ga, 0)

    xlims!(axmain, minval-δ_edge, maxval+δ_edge)
    ylims!(axmain, minval-δ_edge, maxval+δ_edge)


    return fig
end




function quantile_results(
    y,
    ŷ,
    ytest,
    ŷtest,
    varname
    )

    fig = Figure();
    ax = Axis(fig[1,1], xlabel="True $(varname)", ylabel="Predicted $(varname)")

    minval, maxval = extrema([extrema(y)..., extrema(ytest)..., extrema(ŷ)..., extrema(ŷtest)...])
    δ_edge = 0.1*(maxval-minval)

    l1 = lines!(ax, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)
    qtrain = qqplot!(ax, y, ŷ, alpha=0.5)
    qtest = qqplot!(ax, ytest, ŷtest, marker=:rect, alpha=0.5)

    leg = axislegend(ax, [qtrain, qtest, l1], ["Training", "Testing", "1:1"]; position=:lt)

    return fig
end


