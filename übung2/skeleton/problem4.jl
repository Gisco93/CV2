# Load problem2.jl for providing 'load_data'
push!(LOAD_PATH, pwd())
include("problem2.jl");
using Optim
using LineSearches
using Images
using Interpolations

function GAR(x::Array{Float64,2}, alpha::Float64, c::Float64)
    value = 0
    grad = 0
    if(alpha == 2)
        #1/2 (x/c)^2
        value = 0.5 .*(( x ./ c).^2)
        # (x/c)^2
        grad = x ./ (c^2)
    elseif alpha == 0
        #log(1/2 (x/c)^2 + 1)
        value = log.(0.5 .*(( x ./ c).^2) .+ 1.0)
        # 2x/(x^2 + 2* c^2)
        grad = (2.0 .*x) ./ (x.^2 .+ 2.0*c^2)
    elseif alpha == -Inf
        # 1- exp( - 1/2 (x/c)^2)
        value = 1 .- exp.((-0.5) .* ((x ./ c).^2))
        # (x/c)^2 * exp(- 1/2 (x/c)^2)
        grad = (x ./ c^2) .* exp.(-0.5 .*(( x ./ c).^2))
    else
        # abs(alpha - 2) / alpha * (((x / c)^2 / abs(alpha - 2) + 1) ^ (alpha / 2) - 1.0)
        value = (abs(alpha - 2.0) / alpha) .* (((((x./c).^2  ./ abs(alpha - 2.0)) .+ 1.0).^(alpha/2.0)) .- 1.0)
        # (x / c^2) * ((x / c)^2  / abs(alpha - 2.0) + 1.0) ^ (alpha / 2.0))
        grad = (x ./ c^2) .* ((((x./c).^2  ./ abs(alpha - 2.0)) .+ 1.0).^(alpha/2.0 - 1.0))
    end
    value = sum(value)
    return value::Float64, grad::Array{Float64,2}
end


# Evaluate stereo log prior.
# Set: alpha=1.0, sigma=1.0
function stereo_GAR_prior(x::Array{Float64,2}, alpha::Float64, c::Float64)

    height,width = size(x)
    # compute log over vertical and horizontal disparities

    horizontal = GAR(x[:,1:end-1]-x[:,2:end], alpha, c);
    vertical   = GAR(x[1:end-1,:]-x[2:end,:], alpha, c);
    # sum over all vertical and horizontal potentials
    value = horizontal[1] + vertical[1]
    # as the result is 1 row/column short due to indexing... replace this with zeros
    # expand vertical and horizontal to the orignal size and add respective gradient potentials
    grad_h = hcat(horizontal[2], zeros(height,1)) - hcat(zeros(height,1), horizontal[2])
    grad_v = vcat(vertical[2], zeros(1,width)) - vcat(zeros(1,width), vertical[2])
    grad = grad_h .+ grad_v

    return  value::Float64, grad::Array{Float64,2}
end


# Evaluate stereo log likelihood.
# Set: Alpha = 1.0, Sigma = 0.004
function stereo_GAR_likelihood(x::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2}, alpha::Float64, c::Float64)
    # dims
    height,width = size(im0)
    # shift im1 for disparity (need for likelihood)
    im1_x = shift_disparity(im1,x)
    # compute likelihood
    GAR_likelihood = GAR(im0-im1_x, alpha, c);
    # value is than easy:
    value = GAR_likelihood[1]
    # # for the gradient we he have do the Central Differences
    im1_g = 0.5 .* (hcat(zeros(height,2), im1) .- hcat(im1, zeros(height,2)))
    # # and shift this by the disparity
    im1_g_x = shift_disparity(im1_g[:, 2 : end-1], x)
    # Followingly we can use the formula from Lecture 3 Slide 62
    grad = - GAR_likelihood[2] .+ im1_g_x

    return value::Float64, grad::Array{Float64,2}
end

# Evaluate stereo posterior
function stereo_GAR_posterior(x::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2}, alpha::Float64, c::Float64)
    # get prior
    prior = stereo_GAR_prior(x, alpha, c)
    # get likelihood
    likelihood = stereo_GAR_likelihood(x,im0,im1, alpha, c)
    # add values of prior ang likelihood
    log_posterior = prior[1] + likelihood[1]
    # add gradient of prior ang likelihood
    log_posterior_grad = prior[2] .+ likelihood[2]

    return log_posterior::Float64, log_posterior_grad::Array{Float64,2}
end


# Run stereo algorithm using gradient ascent or sth similar
function stereo_GAR(x0::Array{Float64,2}, im0::Array{Float64,2}, im1::Array{Float64,2}, alpha::Float64, c::Float64)
    x = copy(x0);
    #define a value function for Optim
    function value(x)
        return stereo_GAR_posterior(x, im0,im1, alpha, c)[1];
    end
    #define a gradient function for Optim
    function gradient(last, x)
        dx = stereo_GAR_posterior(reshape(x, size(im0)), im0,im1, alpha, c)[2];
        last[:] = dx[:];
    end
    # here we just reused what was used in probem3
    # as results from fitting alpha and c where quite satisfying we didn't change it
    opt = Optim.Options(iterations=500, show_trace=true);#, allow_f_increases=true);
    result = optimize(value, gradient, x0,GradientDescent(linesearch=StrongWolfe()), opt);
    x = reshape(Optim.minimizer(result), size(im0))

    return x::Array{Float64,2}
end
################# Help Functions form Assignment 1 ######################
# Shift all pixels of i1 to the right by the value of gt
# Actually had to edit this Function to be consistent when sampled outside of image
function shift_disparity(i1::Array{Float64,2}, gt::Array{Float64,2})
    id = zeros(size(i1,1),size(i1,2))
    itp = interpolate(i1, BSpline(Linear()))
    itp = extrapolate(itp, 0)
    for i = 1:size(i1,1)
        for j = 1:size(i1,2)
            #apply disparity as shift to the other sides camera if not out side
            if (j - gt[i, j] > 0)
                id[i, j] = itp(i, j-gt[i, j])#i1[i, j-d]
            end
        end
    end
    @assert size(id) == size(i1)
    return id::Array{Float64,2}
end

function shift_disparity2(i1::Array{Float64,2}, gt::Array{Float64,2})
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


function problem4()
    #  Up to you...

    # use problem 2's load_data
    im0, im1, gt = load_data()
    # alpha = 3
    # c = 3

    alpha = 0.25
    c =  0.1
    # #for random
    # alpha = -100.0
    # c = 10.
    # or
    # alpha = 0.05
    # c = 10.0
    ##for constant_disparity with shift_disparity and  allow_f_increases=true in options for GradientDescent:
    # alpha = -Inf
    # c = 100.0
    ##for constant_disparity with shift_disparity2 and allow_f_increases=true in options for GradientDescent:
    # alpha = 0.1
    # c = 0.5
    ## for gt_disparity:
    # alpha = 0.05
    # c = 10.0
    # or
    # alpha = 3.0
    # c = 5.0

    # THIS HURTS MY EYES please specify function input as Tuple{Int64,Int64} as its the return type of the size function
    disparity_size = zeros(Int64, 2,1);
    disparity_size[1] = size(gt,1);
    disparity_size[2] = size(gt,2);
    rand_disparity = random_disparity(disparity_size);
    const_disparity = constant_disparity(disparity_size);
    # # # # Display stereo: Initialized with constant 8's
    result = stereo_GAR(const_disparity, im0, im1,alpha, c);
    result = clamp.(result, 0, 14)
    show_3Plot(result-const_disparity, const_disparity, result, "Diff", "const_disparity", "Opt result")

    # # Display stereo: Initialized with noise in [0,14]
    result = stereo_GAR(rand_disparity, im0, im1,alpha, c);
    result = clamp.(result, 0, 14)
    show_3Plot(result-rand_disparity, rand_disparity, result, "Diff", "rand_disparity", "Opt result")

    # #Display stereo: Initialized with gt
    result = stereo_GAR(gt, im0, im1,alpha, c);
    result = clamp.(result, 0, 14)
    show_3Plot(result-gt, gt, result, "Diff", "gt_disparity", "Opt result")

    ## this is a mess... really dont have a clue how to make coarse too fine here better...
    ## this my seem random but is the best i could find -.-
    ## The idea is to have an adaptive parameter set but no effienct method to optimize it.
    alpha = -100.0
    c = 0.1

    ## Coarse to fine estimation..
    im0_coarse4 = downsample2(downsample2(downsample2(downsample2(im0))))
    im1_coarse4 = downsample2(downsample2(downsample2(downsample2(im1))))
    # gt_coarse4 = downsample2(downsample2(downsample2(downsample2(const_disparity))))
    # gt_coarse4 = downsample2(downsample2(downsample2(downsample2(rand_disparity))))
    gt_coarse4 = downsample2(downsample2(downsample2(downsample2(gt))))
    result_coarse4 = stereo_GAR(gt_coarse4, im0_coarse4, im1_coarse4, alpha, c);
    # show_3Plot(result_coarse4-gt_coarse4, gt_coarse4, result_coarse4, "Diff", "gt coarse16 to fine", "Opt result")
    alpha = -100.0
    c = 0.05
    im0_coarse3 = downsample2(downsample2(downsample2(im0)))
    im1_coarse3 = downsample2(downsample2(downsample2(im1)))
    # not quite sure which resizing algo works better... results are pretty same
    #gt_coarse3 = upsample2(result_coarse4,[3 3])
    gt_coarse3 = Images.imresize(result_coarse4, size(im0_coarse3))
    result_coarse3 = stereo_GAR(gt_coarse3, im0_coarse3, im1_coarse3, alpha, c);
     # show_3Plot(result_coarse4, result_coarse3, result_coarse3, "16", "8", "8")
     alpha = 0.5
     c = 0.05
    im0_coarse2 = downsample2(downsample2(im0))
    im1_coarse2 = downsample2(downsample2(im1))
    #gt_coarse2 = upsample2(result_coarse3,[3 3])
    gt_coarse2 = Images.imresize(result_coarse3, size(im0_coarse2))
    result_coarse2 = stereo_GAR(gt_coarse2, im0_coarse2, im1_coarse2, alpha, c);
    # show_3Plot(result_coarse2-gt_coarse2, gt_coarse2, result_coarse2, "Diff", "gt coarse4 to fine", "Opt result")
    alpha = 0.1
    c = 0.1
    im0_coarse1 = downsample2(im0)
    im1_coarse1 = downsample2(im1)
    #gt_coarse1 = upsample2(result_coarse2,[3 3])
    gt_coarse1 = Images.imresize(result_coarse2, size(im0_coarse1))
    result_coarse1 = stereo_GAR(gt_coarse1, im0_coarse1, im1_coarse1, alpha, c);
    # show_3Plot(result_coarse1-gt_coarse1, gt_coarse1, result_coarse1, "Diff", "gt coarse2 to fine", "Opt result")

    alpha = 3.0
    c = 5.0
    #gt_coarse0 = upsample2(result_coarse1,[3 3])
    gt_coarse0 = Images.imresize(result_coarse1, size(im0))
    result_fine0 = stereo_GAR(gt_coarse0, im0, im1, alpha, c);
    # show_3Plot(result_fine0-gt_coarse0, gt_coarse0, result_fine0, "Diff", "gt fine", "Opt result")

    show_5Plot(result_fine0, result_coarse1, result_coarse2,result_coarse3,result_coarse4, "Opt result", "Opt result/2", "Opt result/4", "Opt result/8", "Opt result/16")



end
