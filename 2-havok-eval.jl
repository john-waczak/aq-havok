using LinearAlgebra
using CairoMakie
using Statistics, StatsBase, Distributions, KernelDensity
using CSV, DataFrames
using Dates, TimeZones
using ProgressMeter

include("./makie-defaults.jl")
set_theme!(mints_theme)
include("./havok.jl")
include("./performance-metrics.jl")

figpath = "./figures/2-havok-eval"
if !ispath(figpath)
    mkpath(figpath)
end

outpath = "./output/2-havok-eval"
if !ispath(outpath)
    mkpath(outpath)
end

df_params = CSV.read("./output/1-havok-pm/param_sweep.csv", DataFrame);

sort(df_params, :rmse_full)[:, [:n_embedding, :r_model, :n_control, :rmse_full, :mae_full]]
# best params: 30, 7, 5


# Load in data
datapath = "./data/data.csv"

df = CSV.read(datapath, DataFrame);
df.datetime .= ZonedDateTime.(String.(df.datetime));

col_to_use = :pm2_5
Zs = df[:, col_to_use]

datetimes = df.datetime
ts = df.dt
ts = ts .+ second(df.datetime[1])
t_days = ts ./ (24*60*60)
d_start = Date(datetimes[1])

# generate nice tick labels
#dt_fmt = dateformat"d-m-yy"
dt_fmt = dateformat"d/m/yy"
Dates.format(d_start, dt_fmt)
tick_labels = [Dates.format(d, dt_fmt) for d ∈ d_start:Day(1):(d_start+Day(10))]
xticks = (0:10, tick_labels)
d_start = Dates.format(d_start, "d/m/yy")

t_days[end]

idx_train = findall(t_days  .≤ 8)
idx_test = findall(t_days .> 8)


Zs_train = Zs[idx_train]
ts_train = ts[idx_train]
t_days_train = t_days[idx_train]

Zs_test = Zs[idx_test]
ts_test = ts[idx_test]
t_days_test = t_days[idx_test]


n_embedding = 30
r_model = 6
n_control = 5
r = r_model + n_control

# Fit the HAVOK model and integrate on test set
Zs_x, Ẑs_x, ts_x, U, σ, V, A, B, fvals = eval_havok(Zs_train, ts_train, n_embedding, r_model, n_control)
Zs_x_test, Ẑs_x_test, ts_x_test = integrate_havok(Zs_test, ts_test, n_embedding, r_model, n_control, A, B, U, σ)


# visualize the fitted time series
c_dgray = colorant"#3d3d3d"
c_dgray = colorant"#737373"
fig = Figure(figure_padding=25);
ax = Axis(fig[2,1],
          xlabel="time (days)",
          ylabel="PM 2.5 (μg/m³)",
          xticks=xticks,
          xminorticks=0:(1/6):10,
          yticks=(-5:5:80),
          xminorgridvisible=true,
          yminorgridvisible=true,
          );

# text!(fig.scene, 0.1, 0.93, text="Training Data", space=:relative, fontsize=15)

vlines!(ax, [2, 2+ 1/6], color=c_dgray, linewidth=1)
vspan!(ax, [2], [2+ 1/6], color=colorant"#ebebeb")

lorig= lines!(ax, ts_x ./ (24*60*60), Zs_x, color=c_dgray, linewidth=2)
lhavok= lines!(ax, ts_x ./ (24*60*60), Ẑs_x, color=mints_colors[3], alpha=0.65, linewidth=1)
lines!(ax, ts_x_test ./ (24*60*60), Zs_x_test, color=c_dgray, linewidth=2)
lhavok_test = lines!(ax, ts_x_test ./ (24*60*60), Ẑs_x_test, color=mints_colors[2], alpha=0.65, linewidth=1)

xlims!(ax, 0, 10)
# ylims!(ax, 0, nothing)
fig[1,1] = Legend(fig, [lorig, lhavok, lhavok_test], ["Original", "HAVOK (training)", "HAVOK (testing)"], framevisible=false, orientation=:horizontal, padding=(0,0,-18,0), labelsize=14, height=-5, halign=:center)

ax_inset = Axis(
    fig[2,1],
    spinewidth=1,
    leftspinecolor=c_dgray,
    rightspinecolor=c_dgray,
    topspinecolor=c_dgray,
    bottomspinecolor=c_dgray,
    width=Relative(0.25),
    height=Relative(0.25),
    halign=0.15,
    valign=0.9,
    backgroundcolor=colorant"#ebebeb",
);
hidedecorations!(ax_inset)
idx_plot = findall(ts_x ./ (24*60*60) .≥ 2 .&& ts_x ./ (24*60*60) .≤ 2 + 1/6 )
lines!(ax_inset, ts_x[idx_plot] ./ (24*60*60), Zs_x[idx_plot], color=c_dgray, linewidth=1)
lines!(ax_inset, ts_x[idx_plot] ./ (24*60*60), Ẑs_x[idx_plot], color=mints_colors[3], alpha=0.65, linewidth=1)
xlims!(ax_inset, ts_x[idx_plot[1]] / (24*60*60), ts_x[idx_plot[end]] / (24*60*60))

translate!(ax_inset.scene, 0, 0, 10)
translate!(ax_inset.elements[:background], 0, 0, 9)
# resize_to_layout!(fig)
fig

save(joinpath(figpath, "1__predicted-ts-training.pdf"), fig)



# visualize the final learned operators:
cmap = cgrad([mints_colors[3], colorant"#FFFFFF", mints_colors[2]], 100)
#crange = (-0.03, 0.03)
crange = round.(extrema(A), sigdigits=2)
fig = Figure();
gl = fig[1,1:2] = GridLayout()
ax1 = Axis(gl[1,1];
           yreversed=true,
           title="A",                  titlesize=25,             titlefont=:regular,
           aspect=DataAspect(),
           xticklabelsvisible=false,   xticksvisible=false,
           xgridvisible=false,         xminorgridvisible=false,
           yticklabelsvisible=false,   yticksvisible=false,
           ygridvisible=false,         yminorgridvisible=false,
           )

ax2 = Axis(gl[1,2];
           yreversed=true,
           title="B",                  titlesize=25,             titlefont=:regular,
           aspect=DataAspect(),
           xticklabelsvisible=false,   xticksvisible=false,
           xgridvisible=false,         xminorgridvisible=false,
           yticklabelsvisible=false,   yticksvisible=false,
           ygridvisible=false,         yminorgridvisible=false,
           )

h1 = heatmap!(ax1, A', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])
h2 = heatmap!(ax2, B', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])

colsize!(gl, 2, Relative(n_control/r))
resize_to_layout!(fig)

fig
# NOTE: need to fix aspect ratio
cb = Colorbar(fig[1,3], colorrange=crange, colormap=cmap, lowclip=mints_colors[3], highclip=mints_colors[2], ticks=range(crange[1], stop=crange[end], length=5), height=Relative(0.7))
fig

save(joinpath(figpath, "2__A-B-Havok.pdf"), fig)



# visualize the eigenvalues of A on complex plane
λs = eigvals(A)

Δreal = maximum(real(λs)) - minimum(real(λs))
Δimag = maximum(imag(λs)) - minimum(imag(λs))

λs[argmax(real(λs))]

# λs
# 6-element Vector{ComplexF64}:
#     -0.007492745974276309 - 0.03522623646234052im
# -0.007492745974276309 + 0.03522623646234052im
# -0.0028086506016698418 - 0.017173847774509973im
# -0.0028086506016698418 + 0.017173847774509973im
# -0.0010098304804486067 + 0.0im
# -0.00019541495815029186 + 0.0im


xx = (1.1*minimum(real(λs)), abs(minimum(real(λs)))*1.1)
yy = (1.1*minimum(imag(λs)), abs(minimum(imag(λs)))*1.1)

c_dblue = colorant"#0e3f57"
fig = Figure();
ax = Axis(fig[1,1,],
          #xlabel="Re(λ)",
          #ylabel="Im(λ)",
          spinewidth=2,
          leftspinecolor=:lightgray, rightspinecolor=:lightgray,
          topspinecolor=:lightgray, bottomspinecolor=:lightgray,
          title="ℂ", titlesize=35, titlealign=:right, titlegap=-5,
          xgridvisible=false, xminorgridvisible=false,
          xticksvisible=false, xticklabelsvisible=false,
          ygridvisible=false, yminorgridvisible=false,
          yticksvisible=false, yticklabelsvisible=false,
          );
#hidespines!(ax)
vlines!(ax, [0.0,], color=:gray, linewidth=3)
hlines!(ax, [0.0,], color=:gray, linewidth=3)
scatter!(ax, real(λs), imag(λs), color=mints_colors[3], strokecolor=c_dblue, strokewidth=2, markersize=15)

text!(ax, 0.675*xx[2], 0.025*yy[2], text="Re(λ)", fontsize=22)
text!(ax, 0.025*xx[2], 0.85*yy[2], text="Im(λ)", fontsize=22)

xlims!(ax, xx[1] ,xx[2])
ylims!(ax, yy[1], yy[2])

colsize!(fig.layout, 1, Aspect(1, 1.0))
resize_to_layout!(fig)

fig

save(joinpath(figpath, "3__A-eigvals.pdf"), fig)



# visualize the forcing statistics
size(fvals)

fig = Figure();
ax = Axis(fig[2,1], yscale=log10, xlabel="f₁", ylabel="p(f₁)");
# ax = Axis(fig[2,1], xlabel="f₁", ylabel="p(f₁)");

forcing_pdf = kde(fvals[:, 1] .- mean(fvals[:, 1])) #, npoints=100)
idxs_nozero = forcing_pdf.density .> 0

# create gaussian using standard deviation of
gauss = Normal(0.0, std(fvals[:, 1]))

l1 = lines!(ax, gauss, linestyle=:dash, linewidth=3, color=mints_colors[2])
l2 = lines!(ax, forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], linewidth=3, color=mints_colors[3])

xlims!(ax, -0.02, 0.02)
ylims!(ax, 10^(-0.8), 10^(2.5))
fig[1,1] = Legend(fig, [l1, l2], ["Gaussian", "Estimated PDF"], framevisible=false, orientation=:horizontal, padding=(0,0,-15,0), labelsize=14)

fig
save(joinpath(figpath, "4__forcing-statistics.pdf"), fig)



# visualize activation of first forcing function
fig = Figure(figure_padding=25);
ax = Axis(fig[1,1], xlabel="time (days)", ylabel="f₁(t)", title="⟨f₁⟩ = $(round(mean(fvals[:,1]), sigdigits=3))", titlealign=:right, titlefont=:regular, titlesize=13, xticks=xticks, xminorticks=1:1/6:20);
lines!(ax, ts_x ./ (24*60*60), fvals[:,1], color=mints_colors[3], linewidth=1)
xlims!(ax, ts_x[1]/(24*60*60), ts_x[end]/(24*60*60))
fig
save(joinpath(figpath, "5__forcing-timeseries.pdf"), fig)



fig = Figure();
ax = Axis(fig[1,1], xlabel="time (minutes)", ylabel="f₁(t)", title="⟨f₁⟩ = $(round(mean(fvals[:,1]), sigdigits=3))", titlealign=:right, titlefont=:regular, titlesize=13) ;
lines!(ax, ts_x[1:1000] ./ (24*60), fvals[1:1000, 1], color=mints_colors[3], linewidth=1)
xlims!(ax, 0, ts_x[1000]/(24*60))
fig

save(joinpath(figpath, "5b_forcing-timeseries-zoomed.pdf"), fig)


# identify forcing active vs not active

thresh = quantile(abs2.(fvals[:,1]), 0.9995)
thresh = 0.0006
forcing_active = zeros(size(fvals, 1))
idx_high = findall(fvals[:,1] .^ 2 .≥thresh)
win = 1000  # max window size to consider forcing
#win = 100  # max window size to consider forcing

forcing_active[idx_high] .= 1

for i ∈ 2:length(idx_high)
    if idx_high[i] - idx_high[i-1] ≤ win
        forcing_active[idx_high[i-1]:idx_high[i]] .= 1
    end
end

lcolor = [fa == 1 ? mints_colors[2] : mints_colors[3] for fa ∈ forcing_active]

fig = Figure(;figure_padding=25);
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="PM 2.5 (μg/m³)", xticksvisible=false, xticklabelsvisible=false,  xticks=1:20);
ax2 = Axis(gl[2,1], ylabel="|f₁|²", xlabel="time (days)", xticks=xticks, xminorticks=1:1/6:20);
linkxaxes!(ax,ax2);

l1 = lines!(ax, ts_x ./ (24*60*60), Zs_x, color=lcolor, linewidth=2)
l2 = lines!(ax2, ts_x ./ (24*60*60), fvals[:,1].^2, linewidth=2, color=lcolor)

rowsize!(gl, 2, Relative(0.2));
xlims!(ax2, ts_x[1]/(24*60*60), ts_x[end]/(24*60*60))

fig

save(joinpath(figpath, "6__timeseries-with-forcing.pdf"), fig)




# create training set for forcing function predictions.
using JSON

out_dict = Dict()

out_dict["A"] = A
out_dict["B"] = B
out_dict["U"] = U
out_dict["σ"] = σ
out_dict["Zs"] = Zs
out_dict["ts"] = ts
out_dict["t_days"] = t_days
out_dict["d_start"] = d_start
out_dict["idx_train"]  = collect(idx_train)
out_dict["idx_test"]  = collect(idx_test)

open("./data/fitres.json", "w") do f
    JSON.print(f, out_dict)
end


