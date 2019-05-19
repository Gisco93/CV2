using PyPlot



# Load Tsukuba disparity dataset
function load_data()

    @assert maximum(gt) <= 16
    return i0::Array{Float64,2}, i1::Array{Float64,2}, gt::Array{Float64,2}
end


# create random disparity in [0,14] of size DISPARITY_SIZE
function random_disparity(disparity_size::Array{Int64,2})



    return disparity_map::Array{Float64,2}
end


# create constant disparity of all 8's of size DISPARITY_SIZE
function constant_disparity(disparity_size::Array{Int64,2})


    return disparity_map::Array{Float64,2}
end


# Evaluate log of Student-t distribution.
# Set sigma=0.7 and alpha=0.8
function log_studentt(x::Array{Float64,2})


    return value::Array{Float64,2}
end

# Evaluate pairwise MRF log prior with Student-t distributions.
# Set sigma=0.7 and alpha=0.8
function mrf_log_prior(x::Array{Float64,2})




    return logp::Float64
end


function problem2()
    i0, i1, gt = load_data()

    # Display log prior of GT disparity map


    # Display log prior of random disparity map


    # Display log prior of constant disparity map



end

