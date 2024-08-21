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

# see https://github.com/sethhirsh/sHAVOK/blob/master/Figure%208.ipynb
# for sHAVOK implementation


figpath = "./figures/2-havok-eval"
if !ispath(figpath)
    mkpath(figpath)
end

outpath = "./output/2-havok-eval"
if !ispath(outpath)
    mkpath(outpath)
end

df_params = CSV.read("./output/1-havok-pm/param_sweep.csv", DataFrame);

dur = "7_day"
sort(df_params[df_params.duration .== dur, [:r_model, :n_control, :n_embedding, :rmse, :mae]], :rmse)
sort(df_params[df_params.duration .== dur .&& df_params.n_control .== 1, [:r_model, :n_control, :n_embedding, :rmse, :mae]], :rmse)


# so the options appear to be (7,5,30) and (7,1,165)

# Load in data
datapath = "./data/df-v1.csv"

df = CSV.read(datapath, DataFrame);
df.datetime .= ZonedDateTime.(String.(df.datetime));
dt_start = df.datetime[1]

t_days = df.dt ./ (60*60*24)


# set up the data
ts = df.dt .- df.dt[1]
dt = ts[2]-ts[1]

idx_train = findall(t_days .≤ 7)
idx_test = findall(t_days .> 7 .&& t_days .≤ 14)

idx_full = findall(t_days .≤ 14)
ts_full = ts[idx_full]

Zs_test = Array(df[idx_test, :pm2_5])
ts_test = ts[idx_test]

Zs = Array(df[idx_train, :pm2_5])
ts = ts[idx_train]


# visualize the results for the final model:
n_embedding = 90
r_model = 5
n_control = 1
r = r_model + n_control

# n_embedding = 165
# r_model = 7
# n_control = 1

r = r_model + n_control

# test out the HAVOK code
Zs_x, Ẑs_x, ts_x, U, σ, V, A, B, fvals = eval_havok(Zs, ts, n_embedding, r_model, n_control)
Zs_x_test, Ẑs_x_test, ts_x_test = integrate_havok(Zs_test, ts_test, n_embedding, r_model, n_control, A, B, U, σ)

# visualize the fitted time series
c_dgray = colorant"#3d3d3d"
c_dgray = colorant"#737373"
fig = Figure();
ax = Axis(fig[2,1],
          xlabel="time (days since $(Date(dt_start)))",
          ylabel="PM 2.5 (μg/m³)",
          xticks=(0:1:15),
          yticks=(0:5:50),
          xminorgridvisible=true,
          yminorgridvisible=true,
          );

text!(fig.scene, 0.1, 0.93, text="Training Data", space=:relative, fontsize=15)

lorig= lines!(ax, ts_x ./ (24*60*60), Zs_x, color=c_dgray, linewidth=2)
lhavok= lines!(ax, ts_x ./ (24*60*60), Ẑs_x, color=mints_colors[3], alpha=0.65, linewidth=1)
xlims!(ax, ts_x[1] / (24*60*60), ts_x[end] / (24*60*60))
ylims!(ax, 0, nothing)
fig[1,1] = Legend(fig, [lorig, lhavok], ["Original", "HAVOK"], framevisible=false, orientation=:horizontal, padding=(0,0,-18,0), labelsize=14, height=-5, halign=:right)
fig

save(joinpath(figpath, "1__predicted-ts-training.pdf"), fig)



c_dgray = colorant"#3d3d3d"
c_dgray = colorant"#737373"
fig = Figure();
ax = Axis(fig[2,1],
          xlabel="time (days since $(Date(dt_start)))",
          ylabel="PM 2.5 (μg/m³)",
          xticks=(0:1:15),
          yticks=(0:5:50),
          xminorgridvisible=true,
          yminorgridvisible=true,
          );

text!(fig.scene, 0.1, 0.93, text="Testing Data", space=:relative, fontsize=15)

lorig= lines!(ax, ts_x_test ./ (24*60*60), Zs_x_test, color=c_dgray, linewidth=2)
lhavok= lines!(ax, ts_x_test ./ (24*60*60), Ẑs_x_test, color=mints_colors[2], alpha=0.65, linewidth=1)
xlims!(ax, ts_x_test[1] / (24*60*60), ts_x_test[end] / (24*60*60))
ylims!(ax, 0, nothing)
fig[1,1] = Legend(fig, [lorig, lhavok], ["Original", "HAVOK"], framevisible=false, orientation=:horizontal, padding=(0,0,-18,0), labelsize=14, height=-5, halign=:right)
fig

save(joinpath(figpath, "1b_predicted-ts-testing.pdf"), fig)


# visualize the final learned operators:
cmap = cgrad([mints_colors[3], colorant"#FFFFFF", mints_colors[2]], 100)
crange = (-0.001, 0.001)
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

save(joinpath(figpath, "3__A-B-Havok.pdf"), fig)



# visualize the eigenvalues of A on complex plane
λs = eigvals(A)

Δreal = maximum(real(λs)) - minimum(real(λs))
Δimag = maximum(imag(λs)) - minimum(imag(λs))

λs[argmax(real(λs))]

# λs
# 5-element Vector{ComplexF64}:
# -0.0007162392071188822 - 0.009066463141579722im
# -0.0007162392071188822 + 0.009066463141579722im
# -0.0002179543630575196 - 0.003729443898762693im
# -0.0002179543630575196 + 0.003729443898762693im
# -4.834513352083636e-7 + 0.0im


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

save(joinpath(figpath, "4__A-eigvals.pdf"), fig)



# visualize the forcing statistics
size(fvals)

fig = Figure();
# ax = Axis(fig[2,1], yscale=log10, xlabel="vᵣ", ylabel="p(vᵣ)");
ax = Axis(fig[2,1], xlabel="vᵣ", ylabel="p(vᵣ)");

forcing_pdf = kde(fvals[:, 1] .- mean(fvals[:, 1])) #, npoints=100)
idxs_nozero = forcing_pdf.density .> 0

# create gaussian using standard deviation of
# gauss = Normal(0.0, std(fvals[:, i]))
gauss = Normal(0.0, std(fvals[:, 1]))

l1 = lines!(ax, gauss, linestyle=:dash, linewidth=3, color=mints_colors[2])
l2 = lines!(ax, forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], linewidth=3, color=mints_colors[3])

# ylims!(0, 120)
#ylims!(10^(0.6), 10^(2.1))
xlims!(-0.02, 0.02)
#xlims!(-0.01, 0.01)

fig[1,1] = Legend(fig, [l1, l2], ["Gaussian", "Estimated PDF"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14)

fig

save(joinpath(figpath, "5__forcing-statistics.pdf"), fig)

# so slightly sharper than Gaussian

# visualize forcing activation on time series
#f_thresh = 0.0001
#c_forcing = [ f ≥ f_thresh ? mints_colors[2] : mints_colors[3] for f ∈ abs2.(fvals[:,1])]
fig = Figure();
gl = fig[1,1] = GridLayout();
ax = Axis(gl[1,1],
          ylabel="PM 2.5 (μg/m³)",
          xticksvisible=false,
          xticklabelsvisible=false,
          xticks=(0:1:8),
          );
axf = Axis(gl[2,1],
           xlabel="time (days since $(Date(dt_start)))",
           ylabel="|f₁|²",
           xticks=(0:1:8),
           yminorgridvisible=false,
           );
linkxaxes!(ax, axf)

l_ts = lines!(ax, ts_x ./ (24*60*60), Zs_x, color=mints_colors[3], linewidth=2)
l_vf = lines!(axf, ts_x ./ (24*60*60), abs2.(fvals[:,1]), color=mints_colors[2], linewidth=2)
#l_vf = lines!(axf, ts_x ./ (24*60*60), abs2.(fvals[:,1]), color=c_forcing, linewidth=2)

rowsize!(gl, 2, Relative(0.2))
xlims!(axf, ts_x[1] / (24*60*60), ts_x[end] / (24*60*60))
fig

save(joinpath(figpath, "6__timeseries-w-forcing.pdf"), fig)


# Vary training length to see impact on performance
ts
ts_test

ts ./ (60*60)

Ltrain = [1:findfirst(ts ./ (60*60) .≥ t) for t ∈ 6:6:(7*24)]

res = []
As = []
Bs = []

@showprogress for i in 1:length(Ltrain)
    L=Ltrain[i]
    ts_train = ts[L]
    Zs_train = Zs[L]

    # fit model
    Zs_x, Ẑs_x, ts_x, U, σ, Vout, A, B, fvals = eval_havok(Zs_train, ts_train, n_embedding, r_model, n_control)
    Zs_x_test, Ẑs_x_test, ts_x_test = integrate_havok(Zs_test, ts_test, n_embedding, r_model, n_control, A, B, U, σ)
    ts_x_test = ts_x_test .- ts_x_test[1]

    push!(As, A)
    push!(Bs, B)

    durations = Dict(
        "1_hr"  => findall(ts_x_test ./ (60*60) .≤ 1),
        "12_hr" => findall(ts_x_test ./ (60*60) .≤ 12),
        "1_day" => findall(ts_x_test ./ (60*60) .≤ 24),
        "2_day" => findall(ts_x_test ./ (60*60) .≤ 2*24),
        "3_day" => findall(ts_x_test ./ (60*60) .≤ 3*24),
        "4_day" => findall(ts_x_test ./ (60*60) .≤ 4*24),
        "5_day" => findall(ts_x_test ./ (60*60) .≤ 5*24),
        "6_day" => findall(ts_x_test ./ (60*60) .≤ 6*24),
        "7_day" => findall(ts_x_test ./ (60*60) .≤ 7*24),
    )

    eval_dict = Dict{String, Any}()
    eval_dict["idx"] = i
    eval_dict["training_dur_days"] = ts[L[end]][end]/(24*60*60)
    for (dur, idxs) ∈ durations
        eval_dict["mae_"*dur] = mean_absolute_error(Ẑs_x_test[idxs], Zs_x_test[idxs])
        eval_dict["rmse_"*dur] = rmse(Ẑs_x_test[idxs], Zs_x_test[idxs])
    end

    push!(res, eval_dict)
end

# turn into CSV and save to output
df_res = DataFrame(res)
sort!(df_res, :training_dur_days)
CSV.write(joinpath(outpath, "duration_sweep.csv"), df_res)
names(df_res)

c_dblue = colorant"#0e3f57"
fig = Figure();
ax = Axis(fig[1,1], xlabel="training period (days)", ylabel="Holdout RMSE (1 day)", xticks=0:1:7);
lines!(df_res.training_dur_days, df_res.rmse_1_day, color=mints_colors[3], linewidth=2)
scatter!(df_res.training_dur_days, df_res.rmse_1_day, color=mints_colors[3], strokecolor=c_dblue, strokewidth=1)
xlims!(ax, 0, 7)

save(joinpath(figpath, "7__dur-vs-error.pdf"), fig)
fig


cmap = cgrad([mints_colors[3], colorant"#FFFFFF", mints_colors[2]], 100)
crange = (-0.005, 0.005)
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

h1 = heatmap!(ax1, As[1]', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])
h2 = heatmap!(ax2, Bs[1]', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])

colsize!(gl, 2, Relative(n_control/r))
resize_to_layout!(fig)

# NOTE: need to fix aspect ratio
cb = Colorbar(fig[1,3], colorrange=crange, colormap=cmap, lowclip=mints_colors[3], highclip=mints_colors[2], ticks=range(crange[1], stop=crange[end], length=5), height=Relative(0.7))
fig
save(joinpath(figpath, "7b_heatmap-dur-0.25.pdf"), fig)


cmap = cgrad([mints_colors[3], colorant"#FFFFFF", mints_colors[2]], 100)
crange = (-0.005, 0.005)
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

h1 = heatmap!(ax1, As[end]', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])
h2 = heatmap!(ax2, Bs[end]', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])

colsize!(gl, 2, Relative(n_control/r))
resize_to_layout!(fig)

# NOTE: need to fix aspect ratio
cb = Colorbar(fig[1,3], colorrange=crange, colormap=cmap, lowclip=mints_colors[3], highclip=mints_colors[2], ticks=range(crange[1], stop=crange[end], length=5), height=Relative(0.7))
fig

save(joinpath(figpath, "7b_heatmap-dur-7.pdf"), fig)


# Vary integration start time to see if error dissappears for large spike in midel of testing dataset


ts_2 = ts[ts ./ (24*60*60) .≥ 1]
Zs_2 = Zs[ts ./ (24*60*60) .≥ 1]

Zs_x, Ẑs_x, ts_x, U, σ, V, A, B, fvals = eval_havok(Zs, ts, n_embedding, r_model, n_control)
Zs_x_2, Ẑs_x_2, ts_x_2= integrate_havok(Zs_2, ts_2, n_embedding, r_model, n_control, A, B, U, σ)

# visualize the fitted time series
c_dgray = colorant"#3d3d3d"
c_dgray = colorant"#737373"
fig = Figure();
ax = Axis(fig[2,1],
          xlabel="time (days since $(Date(dt_start)))",
          ylabel="PM 2.5 (μg/m³)",
          xticks=(0:1:15),
          yticks=(0:5:50),
          xminorgridvisible=true,
          yminorgridvisible=true,
          );

# text!(fig.scene, 0.1, 0.93, text="Training Data", space=:relative, fontsize=15)

lorig= lines!(ax, ts_x ./ (24*60*60), Zs_x, color=c_dgray, linewidth=2)
lhavok= lines!(ax, ts_x ./ (24*60*60), Ẑs_x, color=mints_colors[3], alpha=0.65, linewidth=1)
lhavok2 = lines!(ax, ts_x_2 ./ (24*60*60), Ẑs_x_2, color=mints_colors[2], alpha=0.65, linewidth=1)
xlims!(ax, ts_x[1] / (24*60*60), ts_x[end]/(24*60*60))
ylims!(ax, 0, nothing)
fig[1,1] = Legend(fig, [lorig, lhavok, lhavok2], ["Original", "HAVOK (t₉=0)", "HAVOK (t₀=1)"], framevisible=false, orientation=:horizontal, padding=(0,0,-18,0), labelsize=14, height=-5) #, halign=:right)
fig

save(joinpath(figpath, "8__integrating-later-t0.pdf"), fig)

# So the moral of the story seems to be that the model is stable regardless of where we start the integration, e.g. it's not errors accumulating.


# Try to train model including NO forcing or Single forcing term

# Separately predict the forcing function




