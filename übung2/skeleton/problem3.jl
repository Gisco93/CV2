
# Load problem2.jl for providing 'load_data'
push!(LOAD_PATH, pwd())
include("problem2.jl");


function log_studentt(x::Array{Float64,2}, alpha::Float64, sigma::Float64)




    return value::Float64, grad::Array{Float64,2}
end


# Evaluate stereo log prior.
# Set: alpha=1.0, sigma=1.0
function stereo_log_prior(x::Array{Float64,2})




    return  value::Float64, grad::Array{Float64,2}
end


# Evaluate stereo log likelihood.
# Set: Alpha = 1.0, Sigma = 0.004
function stereo_log_likelihood(x::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2})




    return value::Float64, grad::Array{Float64,2}
end

# Evaluate stereo posterior
function stereo_log_posterior(x::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2})




    return log_posterior::Float64, log_posterior_grad::Array{Float64,2}
end


# Run stereo algorithm using gradient ascent or sth similar
function stereo(x0::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2})




    return x::Array{Float64,2}
end




function problem3()
    # use problem 2's load_data
    im0, im1, gt = load_data()

    # Display stereo: Initialized with constant 8's


    # Display stereo: Initialized with noise in [0,14]


    # Display stereo: Initialized with gt


    # Coarse to fine estimation..


end

