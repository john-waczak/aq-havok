# Function for obtain Hankel Matrix of time-delay embeddings
function Hankel(z, rows)
    cols = length(z) - rows + 1
    H = zeros(rows, cols)

    for i ∈ 1:rows
        H[i,:] .= z[i:i+cols - 1]
    end

    return H
end




