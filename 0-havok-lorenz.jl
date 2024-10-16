using LinearAlgebra
using DifferentialEquations
using DataInterpolations
using CairoMakie
using Statistics, StatsBase, Distributions, KernelDensity




include("./makie-defaults.jl")
set_theme!(mints_theme)
include("./havok.jl")

# see https://github.com/sethhirsh/sHAVOK/blob/master/Figure%208.ipynb
# for sHAVOK implementation


figpath = "./figures/0-havok-lorenz"
if !ispath(figpath)
    mkpath(figpath)
end

outpath = "./output/0-havok-lorenz"
if !ispath(outpath)
    mkpath(outpath)
end




# verify our Hankel matrix implementation works
z = 1:10
Hankel(z, 4)


# Generate data for Lorenz system
σ=10.0
β=8/3
ρ=28.0

p = [σ, ρ, β]

u0 = [-8, 8, 27]
dt = 0.001

ts = range(0, step=dt, length=3000)
# ts = range(0, step=dt, length=300000)

function lorenz!(du, u, p, t)
    x,y,z=u
    σ,ρ,β=p

    du[1] = dx = σ * (y - x)
    du[2] = dy = x * (ρ - z) - y
    du[3] = dz = x * y - β * z
end

prob = ODEProblem(lorenz!, u0, (ts[1], ts[end]), p)
sol = solve(prob, DP5(), saveat=ts, abstol=1e-12, reltol=1e-12);
Data = Array(sol)

# use only the x-component as our measurement for HAVOK
Zs = Data[1,:]
ts
n_embedding = 201
r_model = 14
n_control = 1
r = r_model + n_control

H = Hankel(Zs, n_embedding)

# visualize the learned operators for HAVOK and sHAVOK
cmap = cgrad([mints_colors[3], colorant"#FFFFFF", mints_colors[2]], 100)
crange = (-10, 10)

Ξh,_,_,_ = HAVOK(H, dt, r, 1)
Ah = Ξh[1:r_model, 1:r_model]
Bh = Ξh[1:r_model, r_model+1:r]


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

h1 = heatmap!(ax1, Ah', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])
h2 = heatmap!(ax2, Bh', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])

colsize!(gl, 2, Relative(1/r))
cb = Colorbar(fig[1,3], colorrange=crange, colormap=cmap,  lowclip=mints_colors[3], highclip=mints_colors[2])

fig

save(joinpath(figpath, "1__A-B-Havok.pdf"), fig)



Ξs,_,_,_ = sHAVOK(H, dt, r, 1)
As = Ξs[1:r_model, 1:r_model]
Bs = Ξs[1:r_model, r_model+1:r]


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

h1 = heatmap!(ax1, As', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])
h2 = heatmap!(ax2, Bs', colormap=cmap, colorrange=crange, lowclip=mints_colors[3], highclip=mints_colors[2])

colsize!(gl, 2, Relative(1/r))
cb = Colorbar(fig[1,3], colorrange=crange, colormap=cmap,  lowclip=mints_colors[3], highclip=mints_colors[2])

fig

save(joinpath(figpath, "1b_A-B-sHavok.pdf"), fig)



Ξ,_,_,_ = sHAVOK_central(H, dt, r, 1)
A = Ξs[1:r_model, 1:r_model]
B = Ξs[1:r_model, r_model+1:r]

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

colsize!(gl, 2, Relative(1/r))
cb = Colorbar(fig[1,3], colorrange=crange, colormap=cmap,  lowclip=mints_colors[3], highclip=mints_colors[2])

fig


save(joinpath(figpath, "1c_A-B-sHavok-central.pdf"), fig)




# --------------------------------
#  Integrating the ODE System
# --------------------------------
ts = range(0, step=dt, stop=200)
prob = ODEProblem(lorenz!, u0, (ts[1], ts[end]), p)
sol = solve(prob, DP5(), saveat=ts, abstol=1e-12, reltol=1e-12);
Data = Array(sol)'
Zs = Data[:,1]

fig = Figure();
ga = fig[1,1] = GridLayout();
ax1 = Axis(ga[1,1], ylabel="x", xticklabelsvisible=false, xticksvisible=false)
ax2 = Axis(ga[2,1], ylabel="y", xticklabelsvisible=false, xticksvisible=false)
ax3 = Axis(ga[3,1], xlabel="time", ylabel="z")

linkxaxes!(ax1,ax2,ax3)

lx = lines!(ax1, ts, Data[:,1], color=mints_colors[1])
ly = lines!(ax2, ts, Data[:,2], color=mints_colors[2])
lz = lines!(ax3, ts, Data[:,3], color=mints_colors[3])

xlims!(ax3, 0, 50)
rowgap!(ga, 5)

fig

save(joinpath(figpath, "2__lorenz-xyz-timeseries.pdf"), fig)



# Visualize the original Lorenz attractor
L=findall(ts .< 50.0)
fig = Figure();
ax1 = Axis3(fig[1,1];
            azimuth=-35π/180,
            elevation=30π/180,
            );
hidedecorations!(ax1);
hidespines!(ax1);
l1 = lines!(ax1, Data[L,1], Data[L,2], Data[L,3], color=(:black, 0.65), linewidth=2)

fig

save(joinpath(figpath, "2b_lorenz-attractor.pdf"), fig)


# set up data for fitting HAVOK model
n_embedding = 201
r_model = 14
n_control = 1
r = r_model + n_control

# generate timeseries
Zs = Data[:,1]
ts

Zs_x, Ẑs_x, ts_x, U, σ, Vout, A, B, fvals = eval_havok(Zs, ts, n_embedding, r_model, n_control)

H = Hankel(Zs, n_embedding)
V = H'*U*Diagonal(1 ./ σ)

ts_plot = ts_x[L]


# visualize time-series
fig = Figure();
ga = fig[2,1] = GridLayout();
ax1 = Axis(ga[1,1], ylabel="v₁", xticklabelsvisible=false, xticksvisible=false)
ax2 = Axis(ga[2,1], ylabel="v₂", xticklabelsvisible=false, xticksvisible=false)
ax3 = Axis(ga[3,1], xlabel="time", ylabel="v₃")

linkxaxes!(ax1,ax2,ax3)

l_orig = lines!(ax1, ts_plot, V[L,1], color=mints_colors[3])
lines!(ax2, ts_plot, V[L,2], color=mints_colors[3])
lines!(ax3, ts_plot, V[L,3], color=mints_colors[3])

l_fit = lines!(ax1, ts_plot, Vout[L,1], color=mints_colors[2], alpha=0.7)
lines!(ax2, ts_plot, Vout[L,2], color=mints_colors[2], alpha=0.7)
lines!(ax3, ts_plot, Vout[L,3], color=mints_colors[2], alpha=0.7)

xlims!(ax3, ts_plot[1], ts_plot[end])
rowgap!(ga, 5)

fig[1,1] = Legend(fig, [l_orig, l_fit], ["Original", "HAVOK"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)

fig

save(joinpath(figpath, "3__embedding-timseries-fits.pdf"), fig)



fig = Figure();
ax1 = Axis3(fig[1,1];
            azimuth=-55*π/180,
            elevation=45π/180,
            );
hidedecorations!(ax1);
hidespines!(ax1);
l1 = lines!(ax1, V[L,1], V[L,2], V[L,3], color=(:black, 0.65), linewidth=2)

fig

save(joinpath(figpath, "3b_embedded-attractor.pdf"), fig)


# visualize eigenmodes
fig = Figure();
ax = Axis(fig[1,1], title="Eigenmodes");

ls1 = []
ls2 = []
lr = []
# p = plot([], yticks=[-0.3, 0.0, 0.3], legend=:outerright, label="")

for i ∈ 1:r
    if i ≤ 3
        l = lines!(ax, 1:n_embedding, U[:,i], color=mints_colors[3], linewidth=3)
        push!(ls1, l)
    elseif i > 3 && i < r
        l = lines!(ax, 1:n_embedding, U[:,i], color=:grey, alpha=0.5, linewidth=3)
        push!(ls2, l)
    else
        l = lines!(ax, 1:n_embedding, U[:,i], color=mints_colors[2], linewidth=3)
        push!(lr, l)
    end
end

fig[1,2] = Legend(fig, [ls1..., ls2[1], lr[1]], ["u₁", "u₂", "u₃", "⋮", "uᵣ"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=14, height=-5)

save(joinpath(figpath, "4__U-eigenmodes.pdf"), fig)

fig







# identify forcing active vs not active
thresh = 5e-6
forcing_active = zeros(size(fvals, 1))
idx_high = findall(fvals[:,1] .^ 2 .≥thresh)
win = 500  # max window size to consider forcing

forcing_active[idx_high] .= 1

for i ∈ 2:length(idx_high)
    if idx_high[i] - idx_high[i-1] ≤ win
        forcing_active[idx_high[i-1]:idx_high[i]] .= 1
    end
end

lcolor = [fa == 1 ? mints_colors[2] : mints_colors[1] for fa ∈ forcing_active]

fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="v₁", xticksvisible=false, xticklabelsvisible=false);
ax2 = Axis(gl[2,1], ylabel="|vᵣ|²", xlabel="time");
linkxaxes!(ax,ax2);

l1 = lines!(ax, ts_plot, V[L,1], color=lcolor[L], linewidth=3)
l2 = lines!(ax2, ts_plot, fvals[L,1].^2, linewidth=3, color=lcolor[L])

rowsize!(gl, 2, Relative(0.2));
xlims!(ax2, ts_plot[1], ts_plot[end])

fig

save(joinpath(figpath, "5__v1-with-forcing.pdf"), fig)



# Statistics of forcing function
size(fvals)
forcing_pdf = kde(fvals[:, 1] .- mean(fvals[:, 1]), npoints=100)
idxs_nozero = forcing_pdf.density .> 0

# create gaussian using standard deviation of
gauss = Normal(0.0, std(fvals[:, 1]))

fig = Figure();
ax = Axis(fig[1,1], yscale=log10, xlabel="vᵣ", title="Forcing Statistics");
l1 = lines!(ax, gauss, linestyle=:dash, linewidth=3, color=mints_colors[2])
l2 = lines!(ax, forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], linewidth=3, color=mints_colors[3])
ylims!(10^(-0.5), 1e3)
xlims!(-0.02, 0.02)
fig[1,2] = Legend(fig, [l1, l2], ["Gaussian", "Estimated PDF"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=14)

fig

save(joinpath(figpath, "6__forcing-statistics.pdf"), fig)




# Compute indices where forcing is active
fig = Figure();
ga = fig[1,1] = GridLayout();
ax1 = Axis(ga[1,1], ylabel="v₁", xticklabelsvisible=false, xticksvisible=false)
ax2 = Axis(ga[2,1], ylabel="v₂", xticklabelsvisible=false, xticksvisible=false)
ax3 = Axis(ga[3,1], xlabel="time", ylabel="v₃")

linkxaxes!(ax1,ax2,ax3)

lt = lines!(ax1, ts_plot, V[L,1], color=lcolor[L])
lines!(ax2, ts_plot, V[L,2], color=lcolor[L])
lines!(ax3, ts_plot, V[L,3], color=lcolor[L])
xlims!(ax3, ts_plot[1], ts_plot[end])
rowgap!(ga, 5)

# cb = Colorbar(fig[1,2], lt, ticks=([0, 1e-6, 2e-6, 3e-6, 4e-6], ["0", "1", "2", "3", "4"]), label=rich("Forcing |v", subscript("r"), "|²   ", rich("(×10", superscript("-6"), ")", fontsize=9)))

fig

save(joinpath(figpath, "7__embedding-timeseries-w-forcing.png"), fig, px_per_unit=3)



# Color-code attractor by forcing
fig = Figure();
ax = Axis3(
    fig[1,1];
    azimuth=-35π/180,
    elevation=30π/180,
);

hidespines!(ax)
hidedecorations!(ax)

lines!(ax, Data[L .+ n_embedding,1], Data[L .+ n_embedding,2], Data[L .+ n_embedding,3], color=lcolor[L])

fig

save(joinpath(figpath, "8__attractor-w-forcing.png"), fig, px_per_unit=3)



# reconstruct original time-series
Ĥ = U[:,1:r]*diagm(σ[1:r])*hcat(Vout, fvals)'

Zs_x
Ẑs_x = Ĥ[end,:]

fig = Figure();
ax = Axis(fig[2,1], xlabel="time", ylabel="x(t)");

l_orig = lines!(ax, ts_x[L], Zs_x[L], color=mints_colors[3])
l_havok = lines!(ax, ts_x[L], Ẑs_x[L], color=mints_colors[2], alpha=0.7)

leg = Legend(fig[1,1], [l_orig, l_havok], ["Original", "HAVOK"],framevisible=false, orientation=:horizontal, padding=(0,0,-15,0), labelsize=14)
xlims!(ax, 0, 50)
fig

save(joinpath(figpath, "9__timeseries-reconstruction.pdf"), fig)




# Plot original time series with forcing
fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="x(t)", xticksvisible=false, xticklabelsvisible=false);
ax2 = Axis(gl[2,1], ylabel="|vᵣ|²", xlabel="time");
linkxaxes!(ax,ax2);

l_orig = lines!(ax, ts_x[L], Zs_x[L], color=lcolor[L])
l2 = lines!(ax2, ts_plot, fvals[L,1].^2, linewidth=3, color=lcolor[L])

rowsize!(gl, 2, Relative(0.2));
xlims!(ax2, ts_plot[1], ts_plot[end])

fig

save(joinpath(figpath, "10__og-timeseries-w-forcing.pdf"), fig)





# Now let's see if we can build a model for the forcing function

# visualize the forcing by itself as a single time series
fig = Figure();
ax = Axis(fig[1,1], xlabel="t", ylabel="vᵣ(t)", title="⟨vᵣ⟩ = $(round(mean(fvals[L,1]), sigdigits=3))", titlealign=:right, titlefont=:regular, titlesize=13);
lines!(ax, ts_plot, fvals[L,1], color=mints_colors[3])
xlims!(ax, ts_plot[1], ts_plot[end])
fig

save(joinpath(figpath, "11__forcing-timeseries-w-mean.pdf"), fig)


# okay so it's cleary "mostly" zero centered so we can probably do a decent job of forecasting the forcing function

# start with using the same length of embedding as the time series

# K, b, f_train, f̂_train, f_test, f̂_test = forcing_model(fvals, n_embedding, n_control)
K, f_train, f̂_train, f_test, f̂_test = forcing_model(fvals, n_embedding, n_control)


# visualize scatter diagram of fit
fig = scatter_results(
    f_train[:], f̂_train[:],
    f_test[:], f̂_test[:],
    "vᵣ"
)

save(joinpath(figpath, "12__forcing-model-fit.png"), fig; px_per_unit=2)

