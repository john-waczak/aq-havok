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


figpath = "./figures/1-havok-pm"
if !ispath(figpath)
    mkpath(figpath)
end

outpath = "./output/1-havok-pm"
if !ispath(outpath)
    mkpath(outpath)
end



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

fig = Figure();
ax = Axis(fig[2,1],
          xlabel="time (days since $(Date(dt_start)))",
          ylabel="PM 2.5 (μg/m³)",
          xticks=(0:1:15),
          yticks=(0:5:50),
          xminorgridvisible=true,
          yminorgridvisible=true,
          );

ltrain = lines!(ax, t_days[idx_train], Zs, color=mints_colors[3])
ltest = lines!(ax, t_days[idx_test], Zs_test, color=mints_colors[2])
xlims!(ax, t_days[1], t_days[idx_test[end]])
ylims!(ax, 0, nothing)
fig[1,1] = Legend(fig, [ltrain, ltest], ["Training Data", "Testing Data"], framevisible=false, orientation=:horizontal, padding=(0,0,-18,0), labelsize=14, height=-5, halign=:right)
fig

save(joinpath(figpath, "0__timeseries-pm.pdf"), fig)


# temp = df.temperature[idx_full]
# rh = df.humidity[idx_full]
# dewpoint = temp .- (100 .- rh)./5

# fig = Figure();
# ax = Axis(fig[1,1]);
# lines!(t_days[idx_full], temp .- dewpoint)
# fig


# Evaluate sHAVOK for parameter sweep
n_embeddings = 30:15:(5*60)
rs_model = 3:25
ns_control = 1:5

println("N models: ", length(n_embeddings)*length(rs_model)*length(ns_control))


# duration, mean_bias, mae, mean_bias_norm, mae-norm, rmse, corr_coef, coe, r2
eval_res = []

for i ∈ 1:length(rs_model)
    r_model = rs_model[i]
    for j ∈ 1:length(ns_control)
        n_control = ns_control[j]
        println("r: ", r_model, "\tn: ", n_control)
        @showprogress for k ∈ 1:length(n_embeddings)
            n_embedding = n_embeddings[k]

            Zs_x, Ẑs_x, ts_x, U, σ, V, A, B = eval_havok!(Zs, ts, n_embedding, r_model, n_control)

            durations = Dict(
                "1_hr" => 1:60,
                "12_hr" => 1:12*60,
                "1_day" => 1:24*60,
                "2_day" => 1:2*24*60,
                "3_day" => 1:3*24*60,
                "4_day" => 1:4*24*60,
                "5_day" => 1:5*24*60,
                "6_day" => 1:6*24*60,
                "7_day" => 1:7*24*60,
            )

            for (dur, idxs) in durations
                push!(eval_res, Dict(
                    "n_embedding" => n_embedding,
                    "r_model" => r_model,
                    "n_control" => n_control,
                    "duration" => dur,
                    "mean_bias" => mean_bias(Ẑs_x[idxs], Zs_x[idxs]),
                    "mae" => mean_absolute_error(Ẑs_x[idxs], Zs_x[idxs]),
                    "mean_bias_norm" => normalized_mean_bias(Ẑs_x[idxs], Zs_x[idxs]),
                    "mae_norm" => normalized_mae(Ẑs_x[idxs], Zs_x[idxs]),
                    "rmse" => rmse(Ẑs_x[idxs], Zs_x[idxs]),
                    "corr_coef" => corr_coef(Ẑs_x[idxs], Zs_x[idxs]),
                    "coe" => coefficient_of_efficiency(Ẑs_x[idxs], Zs_x[idxs]),
                    "r2" => r_squared(Ẑs_x[idxs], Zs_x[idxs]),
                ))
            end

        end
    end
end


# turn into CSV and save to output
df_res = DataFrame(eval_res)
CSV.write(joinpath(outpath, "param_sweep.csv"), df_res)




names(df_res)
unique(df_res.duration)
dur = "7_day"

df_res_save = sort(df_res[df_res.duration .== dur .&& df_res.n_control .< df_res.r_model, [:r_model, :n_control, :n_embedding, :rmse, :mae]], :rmse)
CSV.write(joinpath(outpath, "param_sweep_sorted.csv"), df_res_save)

# visualize the results for the final model:
n_embedding = 30
r_model = 7
n_control = 5
r = r_model + n_control

# compute SVD and visualize singular values
Svd = svd(H .- mean(H, dims=1));
σs = abs2.(Svd.S) ./ (size(H,2) - 1)
variances =  σs ./ sum(σs)
cum_var = cumsum(σs ./ sum(σs))


c_outline = colorant"#0e3f57"
fig = Figure();
ax = Axis(fig[2,1], xlabel="Component", ylabel="Explained Variance (%)", xticks=(1:length(σs)), xminorgridvisible=false, yminorgridvisible=false);
barplot!(ax, 1:length(σs), variances .* 100, color=mints_colors[3], strokecolor=c_outline, strokewidth=2)
hl = hlines!(ax, [1,], color=mints_colors[2], linestyle=:solid, linewidth=2, label="1%")
fig[1,1] = Legend(fig, [hl,], ["1 %",], framevisible=false, orientation=:horizontal, padding=(0,0,-20,0), labelsize=14, height=-5, halign=:right)
xlims!(ax, 0.5, 15)
ylims!(ax, -0.01, nothing)
fig

save(joinpath(figpath, "1__pca-explained-variance.pdf"), fig)


# test out the HAVOK code
Zs_x, Ẑs_x, ts_x, U, σ, V, A, B = eval_havok!(Zs, ts, n_embedding, r_model, n_control)

# visualize the final learned operators:
cmap = cgrad([mints_colors[3], colorant"#FFFFFF", mints_colors[2]], 100)
crange = (-0.03, 0.03)

fig = Figure();
gl = fig[1,1:2] = GridLayout()
ax1 = Axis(gl[1,1];
           yreversed=true,
           title="A",                  titlesize=25,             titlefont=:regular,
           xticklabelsvisible=false,   xticksvisible=false,
           xgridvisible=false,         xminorgridvisible=false,
           yticklabelsvisible=false,   yticksvisible=false,
           ygridvisible=false,         yminorgridvisible=false,
           )

ax2 = Axis(gl[1,2];
           yreversed=true,
           title="B",                  titlesize=25,             titlefont=:regular,
           xticklabelsvisible=false,   xticksvisible=false,
           xgridvisible=false,         xminorgridvisible=false,
           yticklabelsvisible=false,   yticksvisible=false,
           ygridvisible=false,         yminorgridvisible=false,
           )

h1 = heatmap!(ax1, A', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])
h2 = heatmap!(ax2, B', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])

colsize!(gl, 2, Relative(n_control/r))
cb = Colorbar(fig[1,3], colorrange=crange, colormap=cmap, lowclip=mints_colors[3], highclip=mints_colors[2], ticks=(-0.03:0.01:0.03))

fig

save(joinpath(figpath, "2__A-B-Havok.pdf"), fig)

# visualize the eigenvalues of A on complex plane

# visualize embedded attractor

# visualize forcing activation on time series

# visualize forcing activation on embedded attractor

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

save(joinpath(figpath, "5__predicted-ts-training.pdf"), fig)

# integrate on test-set  and visualize


# plot model-performance on test set (rmse and mae) (we can decide on the length of integration to use here, e.g. 1day, 7days, etc...)
# versus number of days used in training set.


