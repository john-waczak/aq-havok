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
n_embedding = 30
r_model = 7
n_control = 5
r = r_model + n_control

# test out the HAVOK code
Zs_x, Ẑs_x, ts_x, U, σ, V, A, B = eval_havok!(Zs, ts, n_embedding, r_model, n_control)
Zs_x_test, Ẑs_x_test, ts_x_test = integrate_havok(Zs_test, ts_test, n_embedding, r_model, n_control, A, B, U, σ, V)


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
crange = (-0.03, 0.03)

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
cb = Colorbar(fig[1,3], colorrange=crange, colormap=cmap, lowclip=mints_colors[3], highclip=mints_colors[2], ticks=(-0.03:0.01:0.03), height=Relative(0.7))
fig

save(joinpath(figpath, "3__A-B-Havok.pdf"), fig)

# visualize the eigenvalues of A on complex plane
λs = eigvals(A)

Δreal = maximum(real(λs)) - minimum(real(λs))
Δimag = maximum(imag(λs)) - minimum(imag(λs))

λs[argmax(real(λs))]

# λs
# 7-element Vector{ComplexF64}:
# -0.010307743760888126   -  0.0420881214749148im
# -0.010307743760888126   +  0.0420881214749148im
# -0.004836535492670193   -  0.026294919803872285im
# -0.004836535492670193   +  0.026294919803872285im
# -0.0017037940850779991  -  0.008099762247940904im
# -0.0017037940850779991  +  0.008099762247940904im
# -2.568368768218246e-6   +  0.0im


fig = Figure();
ax = Axis(fig[1,1,],
          #xlabel="Re(λ)",
          #ylabel="Im(λ)",
          spinewidth=2,
          leftspinecolor=:gray, rightspinecolor=:gray,
          topspinecolor=:gray, bottomspinecolor=:gray,
          title="ℂ", titlesize=35, titlealign=:right, titlegap=-5,
          xgridvisible=false, xminorgridvisible=false,
          xticksvisible=false, xticklabelsvisible=false,
          ygridvisible=false, yminorgridvisible=false,
          yticksvisible=false, yticklabelsvisible=false,
          );
#hidespines!(ax)
vlines!(ax, [0.0,], color=:gray, linewidth=3)
hlines!(ax, [0.0,], color=:gray, linewidth=3)
scatter!(ax, real(λs), imag(λs), color=mints_colors[3], markersize=15)
text!(ax, 0.034, 0.001, text="Re(λ)", fontsize=22)
text!(ax, 0.001, 0.0425, text="Im(λ)", fontsize=22)
#text!(ax, -0.035, 0.035, text="ℂ", fontsize=40)
xlims!(ax, -0.05, 0.05)
ylims!(ax, -0.05, 0.05)

# colsize!(fig.layout, 1, Aspect(1, Δreal/Δimag))
colsize!(fig.layout, 1, Aspect(1, 1.0))
resize_to_layout!(fig)

fig

save(joinpath(figpath, "4__A-eigvals.pdf"), fig)



# visualize embedded attractor

# visualize forcing activation on time series

# visualize forcing activation on embedded attractor

# integrate on test-set  and visualize


# plot model-performance on test set (rmse and mae) (we can decide on the length of integration to use here, e.g. 1day, 7days, etc...)
# versus number of days used in training set.


