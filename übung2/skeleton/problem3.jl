
# Load problem2.jl for providing 'load_data'
push!(LOAD_PATH, pwd())
include("problem2.jl");
using Optim;
using LineSearches


function log_studentt(x::Array{Float64,2}, alpha::Float64, sigma::Float64)

    value = sum((-alpha) * log.(1.0 .+ 0.5.*x.*x/(sigma*sigma)))
    grad = - alpha .* x ./ ((sigma*sigma) .+ 0.5* x.*x);

    return value::Float64, grad::Array{Float64,2}
end


# Evaluate stereo log prior.
# Set: alpha=1.0, sigma=1.0
function stereo_log_prior(x::Array{Float64,2})

    height,width = size(x)
    # compute log over vertical and horizontal disparities

    horizontal = log_studentt(x[:,2:end]-x[:,1:end-1],1.0,1.0);
    vertical   = log_studentt(x[2:end,:]-x[1:end-1,:],1.0,1.0);
    # sum over all vertical and horizontal potentials
    value = sum((horizontal[1] + vertical[1]))
    # as the result is 1 row/column short due to indexing... replace this with zeros
    # expand vertical and horizontal to the orignal size and add respective gradient potentials
    grad = [horizontal[2] zeros(height,1)] .+ [vertical[2];zeros(1,width)]

    return  value::Float64, grad::Array{Float64,2}
end


# Evaluate stereo log likelihood.
# Set: Alpha = 1.0, Sigma = 0.004
function stereo_log_likelihood(x::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2})
    # dims
    height,width = size(im0)
    # shift im1 for disparity (need for likelihood)
    im1_x = shift_disparity(im1,x)
    # compute likelihood
    log_likelihood = log_studentt(im0-im1_x, 0.004, 1.0);
    # value is than easy:
    value = log_likelihood[1]
    # for the gradient we he have do the Central Differences
    im1_g = 0.5 * hcat(zeros(height,2), im1) - hcat(im1, zeros(height,2))
    # and shift this by the disparity
    im1_g_x = shift_disparity(im1_g[:, 2 : end-1], x)
    #Followingly we can use the formula from Lecture 3 Slide 62
    grad =  - log_likelihood[2] .* im1_g_x

    return value::Float64, grad::Array{Float64,2}
end

# Evaluate stereo posterior
function stereo_log_posterior(x::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2})

    prior = stereo_log_prior(x)
    likelihood = stereo_log_likelihood(x,im0,im1)
    log_posterior = prior[1] + likelihood[1]
    log_posterior_grad = prior[2] + likelihood[2]

    return log_posterior::Float64, log_posterior_grad::Array{Float64,2}
end


# Run stereo algorithm using gradient ascent or sth similar
function stereo(x0::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2})
    x = copy(x0);

    function f(x)
        return -stereo_log_posterior(reshape(x, size(im0)), im0,im1)[1];
    end

    function g!(storage, x)
        dx = stereo_log_posterior(reshape(x, size(im0)), im0,im1)[2];
        storage[:] = dx[:];
    end

    options = Optim.Options(iterations=500, show_trace=true,allow_f_increases=true);
    result = optimize(f, g!, x, ConjugateGradient(linesearch=StrongWolfe()), options);
    x = reshape(Optim.minimizer(result), size(im0))

    return x::Array{Float64,2}
end
################# Help Functions form Assignment 1 ######################
# Shift all pixels of i1 to the right by the value of gt
function shift_disparity(i1::Array{Float64,2}, gt::Array{Float64,2})
    id = zeros(size(i1,1),size(i1,2))
    for i = 1:size(i1,1)
        for j = 1:size(i1,2)
            #apply disparity as shift to the other sides camera
            id[i,j]= i1[i,round(Int, j-gt[i,j])]
        end
    end
    @assert size(id) == size(i1)
    return id::Array{Float64,2}
end

function show_3Plot(left,right,disparity,title_l, title_r, title_d)
    PyPlot.subplot(1,3,1);
    PyPlot.imshow(left,"gray");
    PyPlot.axis("off");
    PyPlot.title(title_l);
    PyPlot.subplot(1,3,2);
    PyPlot.imshow(right,"gray");
    PyPlot.axis("off");
    PyPlot.title(title_r);
    PyPlot.subplot(1,3,3);
    PyPlot.imshow(disparity,"gray");
    PyPlot.axis("off");
    PyPlot.title(title_d);
    fig1 = PyPlot.gcf();
    display(fig1);
end


function problem3()
    # use problem 2's load_data
    im0, im1, gt = load_data()

    # Display stereo: Initialized with constant 8's
    result = stereo(gt, im0, im1);
    show_3Plot(result-gt, gt, result, "Diff", "Ground Truth", "Opt result")

    # Display stereo: Initialized with noise in [0,14]


    # Display stereo: Initialized with gt


    # Coarse to fine estimation..


end
