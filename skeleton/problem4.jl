# Problem 4: Stereo likelihood

function load_data()

    @assert maximum(gt) <= 16
    return i0::Array{Float64,2}, i1::Array{Float64,2}, gt::Array{Float64,2}
end


# Shift all pixels of i1 to the right by the value of gt
function shift_disparity(i1::Array{Float64,2}, gt::Array{Float64,2})

    @assert size(id) == size(i1)
    return id::Array{Float64,2}
end


# Crop image to the size of the non-zero elements of gt
function crop_image(i::Array{Float64,2}, gt::Array{Float64,2})

    return ic::Array{Float64,2}
end

function make_noise(i::Array{Float64,2}, noise_level::Float64)


    @assert size(i_noise) == size(i)
    return i_noise::Array{Float64,2}
end


# Compute the gaussian likelihood by multiplying the probabilities of a gaussian distribution
# with the given parameters for all pixels
function gaussian_lh(i0::Array{Float64,2}, i1::Array{Float64,2}, mu::Float64, sigma::Float64)


    return l::Float64
end


# Compute the negative logarithmic gaussian likelihood in log domain
function gaussian_nllh(i0::Array{Float64,2}, i1::Array{Float64,2}, mu::Float64, sigma::Float64)


    return nll::Float64
end


# Compute the negative logarithmic laplacian likelihood in log domain
function laplacian_nllh(i0::Array{Float64,2}, i1::Array{Float64,2}, mu::Float64, s::Float64)


    return nll::Float64
end


function problem4()
    # implemented me..
end
