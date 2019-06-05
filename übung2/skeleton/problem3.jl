
# Load problem2.jl for providing 'load_data'
push!(LOAD_PATH, pwd())
include("problem2.jl");
using Optim
using LineSearches




function log_studentt(x::Array{Float64,2}, alpha::Float64, sigma::Float64)

    value = sum(-alpha .* log.(1.0 .+ x.*x ./ (2*sigma*sigma)))
    grad = -2 * alpha .* x ./ (2*sigma*sigma .+ x.*x);

    return value::Float64, grad::Array{Float64,2}
end


# Evaluate stereo log prior.
# Set: alpha=1.0, sigma=1.0
function stereo_log_prior(x::Array{Float64,2})

    height,width = size(x)
    # compute log over vertical and horizontal disparities

    horizontal = log_studentt(x[:,1:end-1]-x[:,2:end],1.0,1.0);
    vertical   = log_studentt(x[1:end-1,:]-x[2:end,:],1.0,1.0);
    # sum over all vertical and horizontal potentials
    value = horizontal[1] + vertical[1]
    # as the result is 1 row/column short due to indexing... replace this with zeros
    # expand vertical and horizontal to the orignal size and add respective gradient potentials
    grad_h = hcat(horizontal[2], zeros(height,1)) - hcat(zeros(height,1), horizontal[2])
    grad_v = vcat(vertical[2], zeros(1,width)) - vcat(zeros(1,width), vertical[2])
    grad = grad_h + grad_v

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
    im1_g = 0.5 * (hcat(zeros(height,2), im1) - hcat(im1, zeros(height,2)))
    # and shift this by the disparity
    im1_g_x = shift_disparity(im1_g[:, 2 : end-1], x)
    #Followingly we can use the formula from Lecture 3 Slide 62
    grad =  - log_likelihood[2] .* im1_g_x

    return value::Float64, grad::Array{Float64,2}
end

# Evaluate stereo posterior
function stereo_log_posterior(x::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2})
    # get prior
    prior = stereo_log_prior(x)
    # get likelihood
    likelihood = stereo_log_likelihood(x,im0,im1)
    # add values of prior ang likelihood
    log_posterior = prior[1] + likelihood[1]
    # add gradient of prior ang likelihood
    log_posterior_grad = prior[2] + likelihood[2]

    return log_posterior::Float64, log_posterior_grad::Array{Float64,2}
end


# Run stereo algorithm using gradient ascent or sth similar
function stereo(x0::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2})
    x = copy(x0);

    function value(x)
        return -stereo_log_posterior(x, im0,im1)[1];
    end

    function gradient(last, x)
        dx = -stereo_log_posterior(reshape(x, size(im0)), im0,im1)[2];
        last[:] = dx[:];
    end
    # till convergance would be nice but ain't nobody got time for that. So max iterations=50
    opt = Optim.Options(iterations=50, show_trace=true);
    # could also be run with linesearch=StrongWolfe() which converges the fastest
    result = optimize(value, gradient, x0,GradientDescent(linesearch=StrongWolfe()), opt);
    x = reshape(Optim.minimizer(result), size(im0))

    return x::Array{Float64,2}
end
################# Help Functions form Assignment 1 ######################
# Shift all pixels of i1 to the right by the value of gt
# Actually had to edit this Function to be consistent when sampled outside of image
function shift_disparity(i1::Array{Float64,2}, gt::Array{Float64,2})
    max_disparity = Int64.(ceil(maximum(abs.(gt))));
    id = zeros(size(i1,1),size(i1,2))
    i1_pad = [zeros(size(i1,1), max_disparity) i1 zeros(size(i1,1), max_disparity)]
    for i = 1:size(i1,1)
        for j = 1:size(i1,2)
            #apply disparity as shift to the other sides camera
            id[i,j]= i1_pad[i, Int64.(round(j-gt[i,j] + max_disparity)) ]
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

function show_5Plot(coarse0,coarse1,coarse2,coarse3,coarse4,title_0, title_1, title_2, title_3, title_4)
    PyPlot.subplot(1,5,1);
    PyPlot.imshow(coarse0,"gray");
    PyPlot.axis("off");
    PyPlot.title(title_0);
    PyPlot.subplot(1,5,2);
    PyPlot.imshow(coarse1,"gray");
    PyPlot.axis("off");
    PyPlot.title(title_1);
    PyPlot.subplot(1,5,3);
    PyPlot.imshow(coarse2,"gray");
    PyPlot.axis("off");
    PyPlot.title(title_2);
    PyPlot.subplot(1,5,4);
    PyPlot.imshow(coarse3,"gray");
    PyPlot.axis("off");
    PyPlot.title(title_3);
    PyPlot.subplot(1,5,5);
    PyPlot.imshow(coarse4,"gray");
    PyPlot.axis("off");
    PyPlot.title(title_4);
    fig1 = PyPlot.gcf();
    display(fig1);
end

################# CV 1 code #########################################
# Create a binomial filter
function makebinomialfilter(size::Array{Int,2})

    ## Calculate weights with Pascals Triangle
    # w^k = pascal triangle
    weights = zeros(size[1])
    for i in 1 : size[1]
        weights[i] = binomial(size[1]-1, i-1)
    end
    # print(weights)

    # Make filter
    f = weights * weights'
    # sum(f) = sum l=0 to N: w^l
    f = f ./ sum(f)
    return f::Array{Float64,2}
end

# Downsample an image by a factor of 2
function downsample2(A::Array{Float64,2})
    D = A[1:2:end, 1:2:end]
  return D::Array{Float64,2}
end

# Upsample an image by a factor of 2
function upsample2(A::Array{Float64,2},fsize::Array{Int,2})

    height2 = 2* size(A)[1]
    width2 = 2* size(A)[2]

    A2 = zeros(height2, width2)

    ## Fill Columns and Rows between old ones with zeros
    for i in 1 : floor(Int, height2/2)
        for j in 1 : floor(Int, width2/2)
            A2[i*2-1,j*2-1] = A[i, j]
            A2[i*2, j*2] = 0
        end
    end

    ## Make binomialfilter with fsize
    f = makebinomialfilter(fsize)

    ## filter A2 with filter
    U = imfilter(A2, f, "symmetric")
    ## apply 4 scale
    U = 4 * U
  return U::Array{Float64,2}
end


function problem3()
    # use problem 2's load_data
    im0, im1, gt = load_data()
    # THIS HURTS MY EYES please specify function input as Tuple{Int64,Int64} as its the return type of the size function
    disparity_size = zeros(Int64, 2,1);
    disparity_size[1] = size(gt,1);
    disparity_size[2] = size(gt,2);
    rand_disparity = random_disparity(disparity_size);
    const_disparity = constant_disparity(disparity_size);
    # # # Display stereo: Initialized with constant 8's
    # result = stereo(const_disparity, im0, im1);
    # show_3Plot(result-const_disparity, const_disparity, result, "Diff", "const_disparity", "Opt result")

    # # Display stereo: Initialized with noise in [0,14]
    # result = stereo(rand_disparity, im0, im1);
    # show_3Plot(result-rand_disparity, rand_disparity, result, "Diff", "rand_disparity", "Opt result")


    # Display stereo: Initialized with gt
    # result = stereo(gt, im0, im1);
    # show_3Plot(result-gt, gt, result, "Diff", "rand_disparity", "Opt result")

    ## Coarse to fine estimation..
    im0_coarse4 = downsample2(downsample2(downsample2(downsample2(im0))))
    im1_coarse4 = downsample2(downsample2(downsample2(downsample2(im1))))
    gt_coarse4 = downsample2(downsample2(downsample2(downsample2(gt))))
    result_coarse4 = stereo(gt_coarse4, im0_coarse4, im1_coarse4);
    # show_3Plot(result_coarse4-gt_coarse4, gt_coarse4, result_coarse4, "Diff", "gt coarse16 to fine", "Opt result")

    im0_coarse3 = downsample2(downsample2(downsample2(im0)))
    im1_coarse3 = downsample2(downsample2(downsample2(im1)))
    gt_coarse3 = upsample2(result_coarse4,[3 3])
    result_coarse3 = stereo(gt_coarse3, im0_coarse3, im1_coarse3);
    # show_3Plot(result_coarse3-gt_coarse3, gt_coarse3, result_coarse3, "Diff", "gt coarse8 to fine", "Opt result")

    im0_coarse2 = downsample2(downsample2(im0))
    im1_coarse2 = downsample2(downsample2(im1))
    gt_coarse2 = upsample2(result_coarse3,[3 3])
    result_coarse2 = stereo(gt_coarse2, im0_coarse2, im1_coarse2);
    # show_3Plot(result_coarse2-gt_coarse2, gt_coarse2, result_coarse2, "Diff", "gt coarse4 to fine", "Opt result")

    im0_coarse1 = downsample2(im0)
    im1_coarse1 = downsample2(im1)
    gt_coarse1 = upsample2(result_coarse2,[3 3])
    result_coarse1 = stereo(gt_coarse1, im0_coarse1, im1_coarse1);
    # show_3Plot(result_coarse1-gt_coarse1, gt_coarse1, result_coarse1, "Diff", "gt coarse2 to fine", "Opt result")

    gt_coarse0 = upsample2(result_coarse1,[3 3])
    result_fine0 = stereo(gt_coarse0, im0, im1);
    # show_3Plot(result_fine0-gt_coarse0, gt_coarse0, result_fine0, "Diff", "gt fine", "Opt result")

    show_5Plot(result_fine0, result_coarse1, result_coarse2,result_coarse3,result_coarse4, "Opt result", "Opt result/2", "Opt result/4", "Opt result/8", "Opt result/16")
end
