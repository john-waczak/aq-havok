using LinearAlgebra
using CairoMakie
using Statistics, StatsBase, Distributions, KernelDensity
using CSV, DataFrames
using Dates, TimeZones
using ProgressMeter
using JSON

include("./makie-defaults.jl")
set_theme!(mints_theme)
include("./havok.jl")
include("./performance-metrics.jl")

figpath = "./figures/3-forcing-fit"
if !ispath(figpath)
    mkpath(figpath)
end

outpath = "./output/3-forcing-fig"
if !ispath(outpath)
    mkpath(outpath)
end

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


# set model parameters
n_embedding = 30
r_model = 6
n_control = 5
r = r_model + n_control

#n_embedding = 105
# r_model = 3
# n_control = 1
# r = r_model + n_control


Zs_x, Ẑs_x, ts_x, U, σ, V, A, B, fvals = eval_havok(Zs_train, ts_train, n_embedding, r_model, n_control)
Zs_x_test, Ẑs_x_test, ts_x_test = integrate_havok(Zs_test, ts_test, n_embedding, r_model, n_control, A, B, U, σ)
fvals_test = (Hankel(Zs_test, n_embedding)'*U*Diagonal(1 ./ σ))[:, r_model+1:r]


fig = Figure();
ax = Axis(fig[1,1]);
lines!(ax, ts_x, Zs_x)
lines!(ax, ts_x, Ẑs_x)
fig

fig = Figure();
ax = Axis(fig[1,1]);
lines!(ax, ts_x_test, Zs_x_test)
lines!(ax, ts_x_test, Ẑs_x_test)
fig

rmse(Ẑs_x, Zs_x)
rmse(Ẑs_x_test, Zs_x_test)


# Model Idea:
# can we predict forcing from the time series itself?
# z_emb + f_now + z_next -> f_next ?
K, fnext, f̂next = get_K_for_forcing(Zs_train, fvals, n_embedding)

rmse(f̂next, fnext)


# now lets check by integrating the model forward
f̂chained = zeros(size(fnext))

Zvec = Hankel(Zs_train, n_embedding)
Zvec_now = Zvec[:, 1:end-1]
z_next = Zvec[end,2:end]
fnow = fvals[1,:]
i_start = 1

for i ∈ i_start:size(f̂chained,1)
    # predict next value
    # i = 1
    Zvec = Zvec_now[:,i]
    znext = z_next[i]

    size(K)

    size(fnow)
    size(znext)
    size(Zvec)

    fpred = K*vcat(fnow, znext, Zvec)

    # update predicted time series
    f̂chained[i,:] .= fpred

    # update fnow
    fnow .= fpred
end

rmse(f̂chained, fnext)

size(fnext)

fig = Figure();
ax = Axis(fig[1,1]);
lines!(ax, 1:size(fnext,1), fnext[:,1])
lines!(ax, 1:size(fnext,1), f̂chained[:,1])
xlims!(ax, 1, 50_000)
fig


# too good to be true?
# RMSE: 2.8675047404657653e-15
# No lets combine this with the HAVOK model to integrate forward in time


# This is recursive forecasting. We could also try direct...
Zs_out, Ẑs_out, ts_out = forecast_havok(Zs_train, ts_train, n_embedding, r_model, n_control, A, B, U, σ, K)

Zs_out[1:10]
Ẑs_out[1:10]

fig = Figure();
ax = Axis(fig[1,1]);
lines!(ax, ts_out, Zs_out)
lines!(ax, ts_out, Ẑs_out)
xlims!(ax, ts_out[1], ts_out[30])
fig

rmse(Ẑs_out[1:30], Zs_out[1:30])

Zs_out, Ẑs_out, ts_out = forecast_havok(Zs_train[1:1+n_embedding+29-1], ts_train[1:n_embedding+29], n_embedding, r_model, n_control, A, B, U, σ, K)


rmse(Ẑs_out, Zs_out)

# evaluate 5-min prediction for multiple starting points

Ztrue_train = []
Zpred_train = []
for i ∈ 1:(length(Zs_train) - (n_embedding+29))
    Zs_out, Ẑs_out, ts_out = forecast_havok(Zs_train[i:i+n_embedding+29-1], ts_train[i:i+n_embedding+29-1], n_embedding, r_model, n_control, A, B, U, σ, K)

    push!(Ztrue_train, Zs_out)
    push!(Zpred_train, Ẑs_out)
end

Ztrue_test = []
Zpred_test = []
for i ∈ 1:(length(Zs_test) - (n_embedding+29))
    Zs_out, Ẑs_out, ts_out = forecast_havok(Zs_test[i:i+n_embedding+29-1], ts_test[i:i+n_embedding+29-1], n_embedding, r_model, n_control, A, B, U, σ, K)

    push!(Ztrue_test, Zs_out)
    push!(Zpred_test, Ẑs_out)
end

# create table for 10 second, 1 minute, 2 minute ... 5 minute  rmse and mae
durs = ["10 sec", "1 min", "2 min", "3 min", "4 min", "5 min"]
idxs = [1, 6, 12, 18, 24, 30]

rmse_train = []
rmse_test = []
mae_train = []
mae_test = []
mape_train = []
mape_test = []
for idx ∈ idxs
    z_true_train = [z[idx] for z ∈ Ztrue_train]
    z_pred_train = [z[idx] for z ∈ Zpred_train]
    z_true_test = [z[idx] for z ∈ Ztrue_test]
    z_pred_test = [z[idx] for z ∈ Zpred_test]

    push!(rmse_train, rmse(z_pred_train, z_true_train))
    push!(rmse_test, rmse(z_pred_test, z_true_test))

    push!(mae_train, mean_absolute_error(z_pred_train, z_true_train))
    push!(mae_test, mean_absolute_error(z_pred_test, z_true_test))

    push!(mape_train, mape(z_pred_train, z_true_train))
    push!(mape_test, mape(z_pred_test, z_true_test))
end

df_res = DataFrame()
df_res.duration = durs
df_res.rmse_train = rmse_train
df_res.rmse_test = rmse_test
df_res.mae_train = mae_train
df_res.mae_test = mae_test
df_res.mape_train = mape_train
df_res.mape_test = mape_test

df_res


CSV.write(joinpath(figpath, "recursive-forecast-res.csv"), df_res)


z_true_train = [z[1] for z ∈ Ztrue_train]
z_pred_train = [z[1] for z ∈ Zpred_train]
z_true_test = [z[1] for z ∈ Ztrue_test]
z_pred_test = [z[1] for z ∈ Zpred_test]

fig = scatter_results(
    z_true_train,
    z_pred_train,
    z_true_test,
    z_pred_test,
    "PM 2.5 (μg/m³)"
)

save(joinpath(figpath, "10-sec-forecast.png"), fig, px_per_unit=2)



z_true_train = [z[6] for z ∈ Ztrue_train]
z_pred_train = [z[6] for z ∈ Zpred_train]
z_true_test = [z[6] for z ∈ Ztrue_test]
z_pred_test = [z[6] for z ∈ Zpred_test]

fig = scatter_results(
    z_true_train,
    z_pred_train,
    z_true_test,
    z_pred_test,
    "PM 2.5 (μg/m³)"
)

save(joinpath(figpath, "1-min-forecast.png"), fig, px_per_unit=2)



z_true_train = [z[12] for z ∈ Ztrue_train]
z_pred_train = [z[12] for z ∈ Zpred_train]
z_true_test = [z[12] for z ∈ Ztrue_test]
z_pred_test = [z[12] for z ∈ Zpred_test]

fig = scatter_results(
    z_true_train,
    z_pred_train,
    z_true_test,
    z_pred_test,
    "PM 2.5 (μg/m³)"
)

save(joinpath(figpath, "2-min-forecast.png"), fig, px_per_unit=2)




z_true_train = [z[end] for z ∈ Ztrue_train]
z_pred_train = [z[end] for z ∈ Zpred_train]
z_true_test = [z[end] for z ∈ Ztrue_test]
z_pred_test = [z[end] for z ∈ Zpred_test]

fig = scatter_results(
    z_true_train,
    z_pred_train,
    z_true_test,
    z_pred_test,
    "PM 2.5 (μg/m³)"
)

save(joinpath(figpath, "5-min-forecast.png"), fig, px_per_unit=2)

