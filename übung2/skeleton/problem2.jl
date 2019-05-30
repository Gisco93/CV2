using PyPlot




# Load Tsukuba disparity dataset
function load_data()
    img0 = PyPlot.imread("i0.png");
    img1 = PyPlot.imread("i1.png");
    i0 = 0.2989 * img0[:,:,1] + 0.5870 * img0[:,:,2] + 0.1140 * img0[:,:,3];
    i1 = 0.2989 * img1[:,:,1] + 0.5870 * img1[:,:,2] + 0.1140 * img1[:,:,3];

    gt = 255.0 * convert(Array{Float64,2},PyPlot.imread("gt.png"));

    @assert maximum(gt) <= 16
    return i0::Array{Float64,2}, i1::Array{Float64,2}, gt::Array{Float64,2}
end


# create random disparity in [0,14] of size DISPARITY_SIZE
function random_disparity(disparity_size::Array{Int64,2})
    # 3 steps:
    # 1: create random floats from 0 to 1 in given size
    # 2: scale floats between [0 14]
    # 3: round floats to Integers  (optional)
    disparity_map = round.(14*rand(disparity_size[1],disparity_size[2]));
    return disparity_map::Array{Float64,2}
end


# create constant disparity of all 8's of size DISPARITY_SIZE
function constant_disparity(disparity_size::Array{Int64,2})
    disparity_map = fill(8.0,(disparity_size[1],disparity_size[2]));
    return disparity_map::Array{Float64,2}
end


# Evaluate log of Student-t distribution.
# Set sigma=0.7 and alpha=0.8
function log_studentt(x::Array{Float64,2})
    value = (-0.8) * log.(1.0 .+ 0.5.*x.*x/0.49)
    return value::Array{Float64,2}
end

# Evaluate pairwise MRF log prior with Student-t distributions.
# Set sigma=0.7 and alpha=0.8
function mrf_log_prior(x::Array{Float64,2})
    height,width = size(x)
    # compute log over vertical and horizontal disparities
    # as the result is 1 row/column short due to indexing... replace this with zeros
    p_horizontal =  [log_studentt(x[:,2:end]-x[:,1:end-1]) zeros(height,1)];
    p_vertical =    [log_studentt(x[2:end,:]-x[1:end-1,:]); zeros(1,width)];
    # sum over all vertcal and horizontal potentials
    # its not nessacary to padd the result back to the Original size
    # as padding wir 0 doesnt change the sum
    logp = sum(p_horizontal +  p_vertical)
    return logp::Float64
end


function problem2()
    i0, i1, gt = load_data();
    # THIS HURTS MY EYES please specify function input as Tuple{Int64,Int64} as its the return type of the size function
    disparity_size = zeros(Int64, 2,1);
    disparity_size[1] = size(gt,1);
    disparity_size[2] = size(gt,2);
    rand_disparity = random_disparity(disparity_size);
    const_disparity = constant_disparity(disparity_size);

    # Display log prior of GT disparity map
    gt_logp = mrf_log_prior(gt)
    println("gt log prior:\t\t\t",gt_logp )
    # Display log prior of random disparity map
    rand_logp  = mrf_log_prior(rand_disparity)
    println("random disparity log prior:\t",rand_logp )
    # Display log prior of constant disparity map
    const_logp = mrf_log_prior(const_disparity)
    println("constant disparity log prior:\t",const_logp )


end
