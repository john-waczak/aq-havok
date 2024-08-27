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


readdir("./data")
# datapath = "./data/df-central-hub-9.csv"
#datapath = "./data/df-valo-node-01.csv"
datapath = "./data/df-central-hub-4.csv"



df = CSV.read(datapath, DataFrame);
nrow(df)
df.datetime .= ZonedDateTime.(String.(df.datetime));

col_to_use = :pm2_5
Zs = df[:, col_to_use]


datetimes = df.datetime
ts = df.dt
ts = ts .+ second(df.datetime[1])
t_days = ts ./ (24*60*60)
d_start = Date(datetimes[1])
t_days[end]


# Visualize all of the data as a single time series
fig = Figure();
ax = Axis(fig[1,1],
          xlabel="time (days since $(Date(d_start)))",
          ylabel="PM 2.5 (μg/m³)",
          #xticks=(0:1:100),
          #yticks=(0:5:55),
          xticks=(0:1:11),
          xminorgridvisible=true,
          yminorgridvisible=true,
          );

lines!(ax, t_days, Zs, color=mints_colors[3])
xlims!(ax, t_days[1], t_days[end])
fig

save(joinpath(figpath, "1__timeseries-full-short.pdf"), fig)


# so split the final set int t_days .≥ 40 for the testing set...
idx_train = findall(t_days  .≤ 8)
#idx_test = findall(t_days .> 7 .&& t_days .≤ 10)
idx_test = findall(t_days .> 8)

Zs_train = Zs[idx_train]
ts_train = ts[idx_train]
t_days_train = t_days[idx_train]

Zs_test = Zs[idx_test]
ts_test = ts[idx_test]
t_days_test = t_days[idx_test]


fig = Figure();
ax = Axis(fig[2,1],
          xlabel="time (days since $(Date(d_start)))",
          ylabel="PM 2.5 (μg/m³)",
          xticks=(0:1:20),
          yticks=(0:5:100),
          xminorgridvisible=true,
          yminorgridvisible=true,
          );

ltrain = lines!(ax, t_days_train, Zs_train, color=mints_colors[3])
ltest = lines!(ax, t_days_test, Zs_test, color=mints_colors[2])
fig[1,1] = Legend(fig, [ltrain, ltest], ["Training Data", "Testing Data"], framevisible=false, orientation=:horizontal, padding=(0,0,-18,0), labelsize=14, height=-5, halign=:right)
xlims!(ax, t_days_train[1], t_days_test[end])
fig

save(joinpath(figpath, "1b__train-test-timeseries-short.pdf"), fig)


# Evaluate sHAVOK for parameter sweep
Δt = ts_train[2] - ts_train[1]
# n_embeddings = 100:10:350
# rs_model = 5:25
# ns_control = 1:5

# n_embeddings = 15:15:(4*60)
# rs_model = 3:25
# ns_control = 1:5

n_embeddings = 30:15:(4*60)
rs_model = 3:25
ns_control = 1:5

triples = [(n_emb, r_mod, n_con) for n_emb ∈ n_embeddings for r_mod ∈ rs_model for n_con ∈ ns_control if n_con < r_mod && r_mod+n_con < n_emb]

println("N models: ", length(triples))
eval_res = []


# @showprogress for k ∈ 1:length(n_embeddings)
@showprogress for triple ∈ triples
    # triple = triples[1]

    n_embedding, r_model, n_control = triple

    Zs_x, Ẑs_x, ts_x, U, σ, _, A, B, _ = eval_havok(Zs_train, ts_train, n_embedding, r_model, n_control)
    # Zs_x, Ẑs_x, ts_x = integrate_havok(Zs_test, ts_test, n_embedding, r_model, n_control, A, B, U, σ)

    ts_x = ts_x .- ts_x[1]

    durations = Dict(
        "1_hr"  => findall(ts_x ./ (60*60) .≤ 1),
        "12_hr" => findall(ts_x ./ (60*60) .≤ 12),
        "1_day" => findall(ts_x ./ (60*60) .≤ 24),
        "full" => findall(ts_x .≤ ts_x[end]),
    )

    res_df = DataFrame()
    res_df.n_embedding = [n_embedding]
    res_df.r_model = [r_model]
    res_df.n_control = [n_control]

    for (dur, idxs) in durations
        res_df[:,"mae_"*dur] = [mean_absolute_error(Ẑs_x[idxs], Zs_x[idxs])]
        res_df[:,"rmse_"*dur] = [rmse(Ẑs_x[idxs], Zs_x[idxs])]
    end

    push!(eval_res, res_df)
end

# turn into CSV and save to output
df_res = vcat(eval_res...)
df_sort = sort(df_res, :rmse_full)[:, [:n_embedding, :r_model, :n_control, :rmse_full, :mae_full]]

# Row │ n_embedding  r_model  n_control  rmse_full  mae_full 
# ──────┼──────────────────────────────────────────────────────
# 1 │          30        6          5   0.216703  0.155912
# 2 │          45       10          5   0.3649    0.268842
# 3 │          30        6          4   0.472797  0.397746
# 4 │          45        8          5   0.487448  0.358938
# 5 │          30        7          5   0.583654  0.518485
# 6 │          30        8          5   0.58551   0.428986
# 7 │          30        7          4   0.586575  0.520145
# 8 │          30        8          4   0.610296  0.447673
# 9 │          30        8          3   0.620928  0.455369
# 10 │          60       12          5   0.697797  0.507499


df_sort = sort(df_res[df_res.n_control .== 1,:], :rmse_full)[:, [:n_embedding, :r_model, :n_control, :rmse_full, :mae_full]]

# Row │ n_embedding  r_model  n_control  rmse_full  mae_full
# ─────┼──────────────────────────────────────────────────────
# 1 │         105        3          1    1.48989   1.09792
# 2 │         120        3          1    1.58473   1.1494
# 3 │         180        5          1    1.77659   1.29506
# 4 │         195        5          1    1.79877   1.29451
# 5 │         135        3          1    1.85822   1.34174
# 6 │         210        5          1    1.9252    1.39189
# 7 │         150        3          1    2.05362   1.49678
# 8 │         225        5          1    2.1544    1.5427
# 9 │          30        8          1    2.24174   1.66401
# 10 │         165        3          1    2.30244   1.67964

CSV.write(joinpath(outpath, "param_sweep.csv"), df_res)


n_embedding = 30
r_model = 6
n_control = 5
Zs_x, Ẑs_x, ts_x, U, σ, _, A, B, fvals = eval_havok(Zs_train, ts_train, n_embedding, r_model, n_control)

fig = Figure();
ax = Axis(fig[1,1]);
lines!(ax, ts_x, Zs_x)
lines!(ax, ts_x, Ẑs_x)
fig


size(fvals)
fig = Figure();
ax = Axis(fig[1,1]);
lines!(ax, ts_x, fvals[:,1])
fig







