using LinearAlgebra

# Function for obtain Hankel Matrix of time-delay embeddings
function Hankel(z, rows)
    cols = length(z) - rows + 1
    H = zeros(rows, cols)

    for i ∈ 1:rows
        H[i,:] .= z[i:i+cols - 1]
    end

    return H
end



# assume Zs is n rows × k features
function Hankel_nd(Zs, n_embedding)
    return hcat([vcat(Zs'[:, i-n_embedding+1:i]...) for i ∈ n_embedding:size(Zs, 1)]...)
end




function true_polys(rows, Δt, r, center=false)
    m = rows ÷ 2  # int division
    p = range(-m*Δt, stop=m*Δt, length=rows)

    U = []
    for j ∈ 1:r
        if center
            push!(U, p .^ (j))
        else
            push!(U, p .^ (j-1))
        end
    end
    U = hcat(U...)

    # Gram-Schmidt to create ONB

    # 1. normalize u₁ to obtain ê₁
    # 2. e₂ = u₂ - (ê₁⋅ u₂)u₂ then normalize to obtain ê₂
    # 3. ...repeat via projection for rest of vectors
    Q = zeros(rows, r)
    for j ∈ 1:r
        v = U[:,j]
        for k ∈ 1:(j-1)
            r_jk = dot(Q[:,k], U[:,j])
            v -= (r_jk * Q[:,k])
        end
        r_jj = norm(v)
        Q[:,j] = v / r_jj
    end

    return Q
end



function HAVOK(H, Δt, r, norm)
    U,σ,V = svd(H)

    # since we know the off-diagonals
    # should be anti-symmetric
    # go through matrix and flip sign
    # using the theoreticl basis

    polys = true_polys(size(H,1), Δt, r)
    for j ∈ 1:r
        if (dot(U[:,j], polys[:,j]) < 0)
            U[:,j] *= -1
            V[:,j] *= -1
        end
    end

    V₁ = V[1:end-1,1:r]
    V₂ = V[2:end,1:r]

    return (V₂'*V₁ - I)/(Δt * norm), U, σ, V
end


# V₁ = V₁[:, 1:r]
# V₂ = V₂[:, 1:r]

# v̇ = Av
# V̇' ≈ (V₂' - V₁')/Δt
#
# use forward Euler
#
# AV₁' = (V₂' - V₁')/Δt
#
# Note: V₁,V₂ are orthogonal
# A = (V₂' - V₁')V₁/(Δt)
#   = (V₂'V₁ - I)/(Δt)
function sHAVOK(H, Δt, r, norm)
    H₁ = H[:, 1:end-1]
    H₂ = H[:, 2:end]

    U₁, σ₁, V₁ = svd(H₁)
    U₂, σ₂, V₂ = svd(H₂)

    # since we know the off-diagonals
    # should be anti-symmetric
    # go through matrix and flip sign
    # using the theoreticl basis
    polys = true_polys(size(H,1), Δt, r)
    for j ∈ 1:r
        if (dot(U₁[:,j], polys[:,j]) < 0)
            U₁[:,j] *= -1
            V₁[:,j] *= -1
        end

        if (dot(U₂[:,j], polys[:,j]) < 0)
            U₂[:,j] *= -1
            V₂[:,j] *= -1
        end
    end

    Vr = V₁[:,1:r]
    Ur = U₁[:,1:r]
    σr = σ₁[1:r]

    return ((V₂'*V₁)[1:r,1:r] - I) / (Δt * norm), Ur, σr, Vr
end


# function sHAVOK_multi(Hs, Δt, r, norm)
#     H1s = [H[:, 1:end-1] for H ∈ Hs]
#     H2s = [H[:, 2:end] for H ∈ Hs]

#     H₁ = hcat(H1s...)
#     H₂ = hcat(H2s...)

#     U₁, σ₁, V₁ = svd(H₁)
#     U₂, σ₂, V₂ = svd(H₂)

#     # since we know the off-diagonals
#     # should be anti-symmetric
#     # go through matrix and flip sign
#     # using the theoreticl basis
#     polys = true_polys(size(H₁,1), Δt, r)
#     for j ∈ 1:r
#         if (dot(U₁[:,j], polys[:,j]) < 0)
#             U₁[:,j] *= -1
#             V₁[:,j] *= -1
#         end

#         if (dot(U₂[:,j], polys[:,j]) < 0)
#             U₂[:,j] *= -1
#             V₂[:,j] *= -1
#         end
#     end

#     Vr = V₁[:,1:r]
#     Ur = U₁[:,1:r]
#     σr = σ₁[1:r]

#     return ((V₂'*V₁)[1:r,1:r] - I) / (Δt * norm), Ur, σr
# end





function sHAVOK_central(H, Δt, r, norm)
    H₁ = H[:, 1:end-2]
    H₂ = H[:, 2:end-1]
    H₃ = H[:, 3:end]

    U₁, σ₁, V₁ = svd(H₁)
    U₂, σ₂, V₂ = svd(H₂)
    U₃, σ₃, V₃ = svd(H₃)

    polys = true_polys(size(H,1), Δt, r)
    for j ∈ 1:r
        if (dot(U₁[:,j], polys[:,j]) < 0)
            U₁[:,j] *= -1
            V₁[:,j] *= -1
        end

        if (dot(U₂[:,j], polys[:,j]) < 0)
            U₂[:,j] *= -1
            V₂[:,j] *= -1
        end

        if (dot(U₃[:,j], polys[:,j]) < 0)
            U₃[:,j] *= -1
            V₃[:,j] *= -1
        end
    end

    Vr = V₂[:,1:r]
    Ur = U₂[:,1:r]
    σr = σ₂[1:r]

    return ((V₃'*V₂)[1:r,1:r] - (V₁'*V₂)[1:r,1:r]) / (Δt * norm), Ur, σr, Vr
end





# solve using Matrix Exponential for each step.
# a single step is given via matrix exponential
#
# | v_next |       | A⋅Δt   B⋅Δt | |    v_now       |
# | f_next | = exp |  0      0  | |    f_now       |
#
#
function make_expM_const(A, B, Δt, r_model, n_control)
    r = r_model + n_control
    M = zeros(r,r)

    M[1:r_model, 1:r_model] .= (A * Δt)
    M[1:r_model, r_model+1:end] .= (B * Δt)
    expM = exp(M)

    expA = expM[1:r_model, 1:r_model]
    expB = expM[1:r_model, r_model+1:end]

    return expA, expB
end



function step_const!(v_next, v_now, f_now, expA, expB)
    v_next .= expA*v_now + expB*f_now
end



# solve using Matrix Exponential for each step.
# a single step is given via matrix exponential
#
# |     v_next     |       | A⋅Δt   B⋅Δt  0 | |    v_now       |
# |     f_next     | = exp |  0      0   I | |    f_now       |
# | f_next - f_now | = exp |  0      0   0 | | f_next - f_now |
#
#
function make_expM_linear(A, B, Δt, r_model, n_control)
    r = r_model + n_control
    M = zeros(r+n_control,r+n_control)

    M[1:r_model, 1:r_model] .= (A * Δt)      # r_model × r_model
    M[1:r_model, r_model+1:r] .= (B * Δt)    # r_model × n_control
    M[r_model+1:r, r+1:end] .= I(n_control)  # r_model × n_control
    expM = exp(M)

    expA = expM[1:r_model, 1:r_model]
    expB = expM[1:r_model, r_model+1:r]
    exp0 = expM[1:r_model, r+1:end]
    return expA, expB, exp0
end


function step_linear!(v_next, v_now, f_next, f_now, expA, expB, exp0)
    v_next .= expA*v_now + expB*f_now + exp0*(f_next .- f_now)
end



function eval_havok(Zs, ts, n_embedding, r_model, n_control)
    r = r_model + n_control

    # construct Hankel Matrix
    H = Hankel(Zs, n_embedding)

    # cutoff time for training vs testing partition
    dt = ts[2]-ts[1]
    Zs_x = H[end, :]
    ts_x = range(ts[n_embedding], step=dt, length=length(Zs_x))

    # compute havok decomposition
    # Ξ,U,σ,V = sHAVOK(H, dt, r, 1);
    Ξ,U,σ,_ = sHAVOK(H, dt, r, 1);

    V = H'*U*Diagonal(1 ./ σ)

    # further truncate original time series to match
    # the data in V
    # Zs_x = Zs_x[1:end-1]
    # ts_x = ts_x[1:end-1]
    Zs_x = Zs_x[1:end]
    ts_x = ts_x[1:end]


    # select Linear and Forcing coef. Matrices
    A = Ξ[1:r_model, 1:r_model];
    B = Ξ[1:r_model, r_model+1:end];

    # set up initial condition
    v₁ = V[1,1:r_model]

    # pick out forcing values
    fvals = V[:,r_model+1:r]

    # construct exponential matrices for time evolution
    expA, expB = make_expM_const(A, B, dt, r_model, n_control)

    # set up outgoing array
    Vout = zeros(size(V, 1), r_model);
    Vout[1,:] .= v₁;
    v_tmp = similar(v₁);

    # compute time evolution
    for i ∈ 2:size(Vout,1)
        step_const!(v_tmp, Vout[i-1,:], fvals[i-1,:], expA, expB)
        Vout[i,:] .= v_tmp
    end

    # reconstruct original time series
    # Ĥ = U*Diagonal(σ)*hcat(Vout, fvals)'
    Ĥ = U[:,1:r_model]*Diagonal(σ[1:r_model])*Vout'
    Ẑs_x = Ĥ[end,:]

    return Zs_x, Ẑs_x, ts_x, U, σ, Vout, A, B, fvals
end


function integrate_havok(Zs, ts, n_embedding, r_model, n_control, A, B, U, σ)
    r = r_model + n_control

    # construct Hankel Matrix
    H = Hankel(Zs, n_embedding)

    # get the current V matrix using U,σ
    # H = UΣV'
    # Σ⁻¹U'H = V'
    # V = H'UΣ⁻¹
    Vcur = H'*U*Diagonal(1 ./ σ)

    # Generate time series via Hankel Matrix
    dt = ts[2]-ts[1]
    Zs_x = H[end, :]
    ts_x = range(ts[n_embedding], step=dt, length=length(Zs_x))

    # set up initial condition
    v₁ = Vcur[1,1:r_model]

    # pick out forcing values
    fvals = Vcur[:,r_model+1:r]


    # construct exponential matrices for time evolution
    expA, expB = make_expM_const(A, B, dt, r_model, n_control)

    # set up outgoing array
    Vout = zeros(size(Vcur, 1), r_model);
    Vout[1,:] .= v₁;
    v_tmp = similar(v₁);

    # compute time evolution
    for i ∈ 2:size(Vout,1)
        step_const!(v_tmp, Vout[i-1,:], fvals[i-1,:], expA, expB)
        Vout[i,:] .= v_tmp
    end

    # reconstruct original time series
    # Ĥ = U*Diagonal(σ)*hcat(Vout, fvals)'
    Ĥ = U*Diagonal(σ)*hcat(Vout, fvals)'
    Ĥ = U[:,1:r_model]*Diagonal(σ[1:r_model])*Vout'
    Ẑs_x = Ĥ[end,:]

    return Zs_x, Ẑs_x, ts_x
end



get_v_from_z(z, invΣ, U) = invΣ*U'*z
get_z_from_v(v, Σ, U, r_model) = U[:, 1:r_model]*Σ[1:r_model, 1:r_model]*v[1:r_model]
get_z_from_v_and_f(v, f, Σ, U, r) = U[:, 1:r]*Σ[1:r, 1:r]*vcat(v,f)



function integrate_havok_nonneg(Zs, ts, n_embedding, r_model, n_control, A, B, U, σ)
    r = r_model + n_control

    # construct Hankel Matrix
    H = Hankel(Zs, n_embedding)

    # get the current V matrix using U,σ
    # H = UΣV'
    # Σ⁻¹U'H = V'
    # V = H'UΣ⁻¹
    Vcur = H'*U*Diagonal(1 ./ σ)

    # Generate time series via Hankel Matrix
    dt = ts[2]-ts[1]
    Zs_x = H[end, :]
    ts_x = range(ts[n_embedding], step=dt, length=length(Zs_x))

    # set up initial condition
    v₁ = Vcur[1,1:r_model]

    # pick out forcing values
    fvals = Vcur[:,r_model+1:r]


    # construct exponential matrices for time evolution
    expA, expB = make_expM_const(A, B, dt, r_model, n_control)

    # set up outgoing array
    Vout = zeros(size(Vcur, 1), r_model);
    Vout[1,:] .= v₁;
    v_tmp = similar(v₁);

    # compute time evolution
    for i ∈ 2:size(Vout,1)
        step_const!(v_tmp, Vout[i-1,:], fvals[i-1,:], expA, expB)
        Vout[i,:] .= v_tmp
    end

    # reconstruct original time series
    # Ĥ = U*Diagonal(σ)*hcat(Vout, fvals)'
    Ĥ = U*Diagonal(σ)*hcat(Vout, fvals)'
    Ĥ = U[:,1:r_model]*Diagonal(σ[1:r_model])*Vout'
    Ẑs_x = Ĥ[end,:]

    return Zs_x, Ẑs_x, ts_x
end


using Random

function forcing_model(fvals, n_embedding, n_control)
    fs_emb = Hankel_nd(fvals, n_embedding)

    fs = fs_emb[end - n_control + 1 : end,:]

    F = fs_emb[:, 1:end-1]
    f_next = fs[:, 2:end]

    # verify we have shifted things correctly
    @assert all(F[end - n_control + 1,2] .== f_next[:,1])


    size(f_next)
    ntrain = Int(0.8*size(f_next,2))
    idx_train = shuffle(1:size(f_next,2))[1:ntrain]
    idx_test = setdiff(1:size(f_next,2), idx_train)

    Ftrain = F[:, idx_train]
    Ftest = F[:, idx_test]
    f_train = f_next[:, idx_train]
    f_test = f_next[:, idx_test]




    # M = f_train / vcat(Ftrain, ones(size(f_train)))

    K = f_train / Ftrain

    # K = M[:, 1:end-1]
    # b = M[:, end]

    # now create predictions for forcing function
    # f̂_train = (K*Ftrain .+ b)
    # f̂_test = (K*Ftest .+ b)

    f̂_train = K*Ftrain
    f̂_test = K*Ftest



    # return K, b, f_train, f̂_train, f_test, f̂_test
    return K, f_train, f̂_train, f_test, f̂_test
end


# step_forcing_model(f_emb, K, b) = K*f_emb .+ b


# function forecast_havok(Zs, ts, n_embedding, r_model, n_control, A, B, U, σ)
#     dt = ts[2]-ts[1]

#     r = r_model + n_control

#     # get first embedding vector for time delay
#     V_0 = Diagonal(1 ./ σ)*U'*Zs[1:n_embedding]

#     # set up initial condition
#     v_prev = V_0[1:r_model]
#     f_prev = V_0[r_model+1:r]

#     # construct exponential matrices for time evolution
#     expA, expB = make_expM_const(A, B, dt, r_model, n_control)


#     # set up outgoing array
#     v_curr = similar(v_prev);      # for updating state
#     f_curr = similar(f_prev);      # for updating forcing

#     # set up outgoing array for time series predictions
#     Zs_out = zeros(length(Zs))
#     Zs_out[1:n_embedding] .= Zs[1:n_embedding]

#     # pre-compute this for re-use
#     σu = σ[1:r_model] .* U[end,1:r_model]
#     ΣinvUt = Diagonal(1 ./ σ[r_model+1:r]) * U[:, r_model+1:r]'

#     # compute time evolution
#     for i ∈ n_embedding+1:length(Zs)
#         # step the model forward
#         step_const!(v_curr, v_prev, f_prev, expA, expB)

#         # update the state
#         v_prev .= v_curr

#         # update next time series value
#         Zs_out[i] = dot(σu, v_curr)

#         # use state to estimate forcing
#         # mul!(f_curr, ΣinvUt, Zs_out[i-n_embedding+1:i])
#         f_curr =  ΣinvUt * Zs_out[i-n_embedding+1:i]

#         # update forcing
#         f_prev = f_curr
#     end

#     return Zs_out
# end


