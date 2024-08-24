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
readdir("./data")
#datapath = "./data/df-polo-node-21.csv"  # UT Dallas ECSN
# datapath = "./data/df-valo-node-01.csv"  # IQHQ
# datapath = "./data/df-valo-node-04.csv"  # Bedford
# datapath = "./data/df-polo-node-23.csv"

datapath = "./data/df-central-hub-9.csv"  # Paul Quinn


df = CSV.read(datapath, DataFrame);
df.datetime .= ZonedDateTime.(String.(df.datetime));

col_to_use = :pm2_5
Zs = df[:, col_to_use]


# findall(Zs .== 0.0)

datetimes = df.datetime
ts = df.dt
ts = ts .+ minute(df.datetime[1])
t_days = ts ./ (24*60*60)
d_start = Date(datetimes[1])

t_days[end]

# Visualize all of the data as a single time series, alternating colors
# Between gaps
fig = Figure();
ax = Axis(fig[1,1],
          xlabel="time (days since $(Date(d_start)))",
          ylabel="PM 2.5 (μg/m³)",
          xticks=(0:1:100),
          #yticks=(0:5:55),
          xminorgridvisible=true,
          yminorgridvisible=true,
          );

lines!(ax, t_days, Zs, color=mints_colors[3])
xlims!(ax, t_days[1], t_days[end])
#ylims!(ax, -0.1, 50)
fig

save(joinpath(figpath, "0__timeseries-full.pdf"), fig)


# so split the final set int t_days .≥ 40 for the testing set...

idx_train = findall(t_days .≤ 7)
idx_test = findall(t_days .> 7 .&& t_days .≤ 14)

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
          xticks=(0:1:15),
          yticks=(0:5:50),
          xminorgridvisible=true,
          yminorgridvisible=true,
          );

ltrain = lines!(ax, t_days_train, Zs_train, color=mints_colors[3])
ltest = lines!(ax, t_days_test, Zs_test, color=mints_colors[2])
fig[1,1] = Legend(fig, [ltrain, ltest], ["Training Data", "Testing Data"], framevisible=false, orientation=:horizontal, padding=(0,0,-18,0), labelsize=14, height=-5, halign=:right)
xlims!(ax, t_days_train[1], t_days_test[end])
#ylims!(ax, -0.1, 50)
fig

save(joinpath(figpath, "0b__train-test-timeseries.pdf"), fig)

# Evaluate sHAVOK for parameter sweep
Δt = ts_train[2] - ts_train[1]

n_embeddings = 30:15:120
rs_model = 3:40
ns_control = 0:15


triples = [(n_emb, r_mod, n_con) for n_emb ∈ n_embeddings for r_mod ∈ rs_model for n_con ∈ ns_control if n_con < r_mod && r_mod+n_con < n_emb]

println("N models: ", length(triples))
eval_res = []

# @showprogress for k ∈ 1:length(n_embeddings)
@showprogress for triple ∈ triples
    n_embedding, r_model, n_control = triple

    Zs_x, Ẑs_x, ts_x, U, σ, _, A, B, _ = eval_havok(Zs_train, ts_train, n_embedding, r_model, n_control)
    # Zs_x, Ẑs_x, ts_x = integrate_havok(Zs_test, ts_test, n_embedding, r_model, n_control, A, B, U, σ)
    ts_x = ts_x .- ts_x[1]

    durations = Dict(
        "1_hr"  => findall(ts_x ./ (60*60) .≤ 1),
        "12_hr" => findall(ts_x ./ (60*60) .≤ 12),
        "1_day" => findall(ts_x ./ (60*60) .≤ 24),
        "2_day" => findall(ts_x ./ (60*60) .≤ 2*24),
        "3_day" => findall(ts_x ./ (60*60) .≤ 3*24),
        "4_day" => findall(ts_x ./ (60*60) .≤ 4*24),
        "5_day" => findall(ts_x ./ (60*60) .≤ 5*24),
        "6_day" => findall(ts_x ./ (60*60) .≤ 6*24),
        "7_day" => findall(ts_x ./ (60*60) .≤ 7*24),
        "10_day" => findall(ts_x ./ (60*60) .≤ 10*24),
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
# df_res = DataFrame(eval_res)
df_res = vcat(eval_res...)
df_sort = sort(df_res, :rmse_10_day)[:, [:n_embedding, :r_model, :n_control, :rmse_10_day, :mae_10_day]]

# Row │ n_embedding  r_model  n_control  rmse_10_day  mae_10_day 
# ──────┼──────────────────────────────────────────────────────────
# 1 │          45       11         10     0.249638    0.199078
# 2 │          75       17         11     0.326801    0.258223
# 3 │          75       23         14     0.32749     0.260647
# 4 │          75       23         13     0.3275      0.260655
# 5 │          75       23         15     0.327513    0.260668
# 6 │          75       25         15     0.33929     0.270394
# 7 │          75       25         14     0.341107    0.271865
# 8 │          75       25         12     0.341645    0.272309
# 9 │          75       25         11     0.341654    0.272317
# 10 │          75       25         13     0.341716    0.272369


df_sort = sort(df_res[df_res.n_control .== 1,:], :rmse_10_day)[:, [:n_embedding, :r_model, :n_control, :rmse_10_day, :mae_10_day]]

# Row │ n_embedding  r_model  n_control  rmse_10_day  mae_10_day 
# ─────┼──────────────────────────────────────────────────────────
# 1 │          90       11          1     0.779965    0.612343
# 2 │          45        3          1     0.798096    0.6352
# 3 │          75        7          1     0.803061    0.64456
# 4 │          60        3          1     0.929936    0.732968
# 5 │          90        5          1     0.974242    0.807959
# 6 │         105        9          1     0.991745    0.777937
# 7 │          60        5          1     1.17818     0.946626
# 8 │          30        3          1     1.19414     0.986072
# 9 │          75       17          1     1.19866     0.982219
# 10 │         105       13          1     1.21488     0.978601


CSV.write(joinpath(outpath, "param_sweep.csv"), df_res)


n_embedding = 45
r_model = 11
n_control = 10

Zs_x, Ẑs_x, ts_x, U, σ, _, A, B, _ = eval_havok(Zs_train, ts_train, n_embedding, r_model, n_control)



fig = Figure();
ax = Axis(fig[1,1]);
lines!(ax, ts_x ./ (24*60*60), Zs_x)
lines!(ax, ts_x ./ (24*60*60), Ẑs_x)
#ylims!(ax, 0, 50)
fig

Zs_x, Ẑs_x, ts_x = integrate_havok(Zs_test, ts_test, n_embedding, r_model, n_control, A, B, U, σ)
ts_x = ts_x .- ts_x[1]

fig = Figure();
ax = Axis(fig[1,1]);
lines!(ax, ts_x ./ (24*60*60), Zs_x)
lines!(ax, ts_x ./ (24*60*60), Ẑs_x)
#ylims!(ax, 0, 50)
fig


# compute SVD and visualize singular values
H = Hankel(Zs, n_embedding)
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
xlims!(ax, 0.5, 10.5)
ylims!(ax, -0.01, nothing)
fig


save(joinpath(figpath, "1__pca-explained-variance.pdf"), fig)


# Compare forcing functions to see if they linearly related...

# H = Hankel(Zs_train, n_embedding)
# V = H'*U*Diagonal(1 ./ σ)

# fig = Figure();
# ax = Axis(fig[1,1]);
# scatter!(ax, V[:, r_model+1], V[:, r_model+2])
# fig




