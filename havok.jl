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
    Ξ,U,σ,V = sHAVOK(H, dt, r, 1);

    # further truncate original time series to match
    # the data in V
    Zs_x = Zs_x[1:end-1]
    ts_x = ts_x[1:end-1]

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

    # cutoff time for training vs testing partition
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




function forecast_havok(Zs, ts, n_embedding, r_model, n_control, A, B, U, σ)
    dt = ts[2]-ts[1]

    r = r_model + n_control

    # get first embedding vector for time delay
    V_0 = Diagonal(1 ./ σ)*U'*Zs[1:n_embedding]

    # set up initial condition
    v_prev = V_0[1:r_model]
    f_prev = V_0[r_model+1:r]

    # construct exponential matrices for time evolution
    expA, expB = make_expM_const(A, B, dt, r_model, n_control)


    # set up outgoing array
    v_curr = similar(v_prev);      # for updating state
    f_curr = similar(f_prev);      # for updating forcing

    # set up outgoing array for time series predictions
    Zs_out = zeros(length(Zs))
    Zs_out[1:n_embedding] .= Zs[1:n_embedding]

    # pre-compute this for re-use
    σu = σ[1:r_model] .* U[end,1:r_model]
    ΣinvUt = Diagonal(1 ./ σ[r_model+1:r]) * U[:, r_model+1:r]'

    # compute time evolution
    for i ∈ n_embedding+1:length(Zs)
        # step the model forward
        step_const!(v_curr, v_prev, f_prev, expA, expB)

        # update the state
        v_prev .= v_curr

        # update next time series value
        Zs_out[i] = dot(σu, v_curr)

        # use state to estimate forcing
        # mul!(f_curr, ΣinvUt, Zs_out[i-n_embedding+1:i])
        f_curr =  ΣinvUt * Zs_out[i-n_embedding+1:i]

        # update forcing
        f_prev = f_curr
    end

    return Zs_out
end




# reconstruct!(z_emb, v_next, Uσ) = mul!(z_emb, Uσ, v_next)


# function step!(v_next, v_now, v_prev, f_now, A, B, Δt)
#     # use a central difference scheme
#     v_next  .= v_prev
#     v_next .+= (2*Δt) *  (A*v_now + B*f_now)
# end





# # compute current forcing function value using reconstructed
# function forcing(v_full, U, σ, r)
#     vr_curr = (1/σ[r])*dot(U[:,r],v_full)
# end



