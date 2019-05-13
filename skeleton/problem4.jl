using Distributions
using Random
import PyPlot
# Problem 4: Stereo likelihood

function load_data()
    img0 = PyPlot.imread("i0.png")
    img1 = PyPlot.imread("i1.png")
    i0 = 0.2989 * img0[:,:,1] + 0.5870 * img0[:,:,2] + 0.1140 * img0[:,:,3];
    i1 = 0.2989 * img1[:,:,1] + 0.5870 * img1[:,:,2] + 0.1140 * img1[:,:,3];

    gt = 255 * channelview(float64.(load("gt.png")));
    @assert maximum(gt) <= 16
    return i0::Array{Float64,2}, i1::Array{Float64,2}, gt::Array{Float64,2}
end


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


# Crop image to the size of the non-zero elements of gt
function crop_image(i::Array{Float64,2}, gt::Array{Float64,2})
    # See https://docs.julialang.org/en/v1.1/base/arrays/#Views-(SubArrays-and-other-view-types)-1
    # findall gives all index entries for which gt is not zero,
    not_zeros = findall(!iszero, gt);
    # select minimum/maximum for rows and columns of image i
    ic = i[minimum(not_zeros)[1]:maximum(not_zeros[:])[1],minimum(not_zeros)[2]:maximum(not_zeros[:])[2]];
    return ic::Array{Float64,2}
end

function make_noise(i::Array{Float64,2}, noise_level::Float64)
    i_noise = deepcopy(i)
    dim_1 = size(i,1)
    dim_2 = size(i,2)

    # Get all keys(Cartesian Coords) from i and shuffle to introduce randomness
    idx_img = Random.shuffle(keys(i));
    # exact number of noisy pixels rounded
    num_rand = round(Int,noise_level*dim_1*dim_2)
    # replace randomly selected pixels with more randomness ;)
    for num_i = 1:num_rand
        # sample randomly between [0.1,0.9]
        i_noise[idx_img[num_i]] = 0.8*rand()+ 0.1
    end

    @assert size(i_noise) == size(i)
    return i_noise::Array{Float64,2}
end


# Compute the gaussian likelihood by multiplying the probabilities of a gaussian distribution
# with the given parameters for all pixels
function gaussian_lh(i0::Array{Float64,2}, i1::Array{Float64,2}, mu::Float64, sigma::Float64)
    scaling = 1/(sqrt(2*pi*sigma^2 ))
    l=1
    for i = 1:size(i0,1)
        for j = 1:size(i0,2)
            # Likelyhood:
            #l *= scaling
            l *=  exp(-((((i0[i,j]-i1[i,j])-mu)^2)/(2*sigma^2)))
        end
    end
    return l::Float64
end


# Compute the negative logarithmic gaussian likelihood in log domain
function gaussian_nllh(i0::Array{Float64,2}, i1::Array{Float64,2}, mu::Float64, sigma::Float64)
    scaling = (sqrt(2*pi*sigma^2 ))
    nll=0
    for i = 1:size(i0,1)
        for j = 1:size(i0,2)
            #log: -log(scaling) -((((i0[i,j]-i1[i,j])-mu)^2)/(2*sigma^2))
            #negative log:
            #nll += log(scaling)
            nll += ((i0[i,j]-i1[i,j])-mu)^2
        end
    end
    nll = nll/(2*sigma^2)
    return nll::Float64
end


# Compute the negative logarithmic laplacian likelihood in log domain
function laplacian_nllh(i0::Array{Float64,2}, i1::Array{Float64,2}, mu::Float64, s::Float64)
    scaling = 2*s
    nll = 0
    for i = 1:size(i0,1)
        for j = 1:size(i0,2)
            #log: -log(scaling) -((((i0[i,j]-i1_d[i,j])-mu)^2)/(2*s^2))
            #negative log:
            #nll += log(scaling)
            nll += abs(i0[i,j]-i1[i,j]-mu)
        end
    end
    nll = nll/s
    return nll::Float64
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

function problem4()
    # implemented me..
    #given mu & sigma & s
    mu = 0.0
    sigma = 1.2
    s = 1.2
    # load I0, I1 and d_gt
    img0,img1,d = load_data()
    #show_3Plot(img0,img1,d,"Left","Right","Disparity");
    #2) apply disparity and crop
    println("2):\n -Shift img1 with disparity \n -Crop images \n -Compute gaussian Log Likelyhood")
    #apply Disparity to img1
    img1_d = shift_disparity(img1,d)
    #crop all images and d
    img0_crop = crop_image(img0,d)
    img1_d_crop = crop_image(img1_d,d)
    d_crop = crop_image(d,d)
    #show_3Plot(img0_crop,img1_d_crop,d_crop,"Left cropped","Right shifted & cropped","Disparity cropped");
    # Gausian Log Likelyhood
    # DISCLAIMER as for part 6 the scaling terms are commented out.
    # This is for better compareability between the Gausian and Laplacian Likelyhood
    # as the scaling terms vary and (seemingly) migate the effect of noise for the Gausian.
    # search for: "scaling" to comment them in as you need, please.
    println("Gausian lh: ", gaussian_lh(img0_crop,img1_d_crop,mu,sigma))
    println("3): \n Compute gaussian negative Log Likelyhood")
    println("Gausian nllh: ", gaussian_nllh(img0_crop,img1_d_crop,mu,sigma))

    #4) make images with noise...
    # as they also need to be shifted and cropped, we'll use img1_d_crop for simplification
    img1_n12 = make_noise(img1_d_crop, 0.12);
    img1_n25 = make_noise(img1_d_crop, 0.25);

    #show_3Plot(img1_d_crop,img1_n12,img1_n25,"Original","12% Noise","25% Noise");
    println("4): \n Compute gaussian Likelyhood for 12% noise")
    println("Gausian lh: ", gaussian_lh(img0_crop,img1_n12,mu,sigma))
    println("Compute gaussian negative Log Likelyhood for 12% noise")
    println("Gausian nllh: ", gaussian_nllh(img0_crop,img1_n12,mu,sigma))

    println("Compute gaussian Likelyhood for 25% noise")
    println("Gausian lh: ", gaussian_lh(img0_crop,img1_n25,mu,sigma))
    println("Compute gaussian negative Log Likelyhood for 25% noise")
    println("Gausian nllh: ", gaussian_nllh(img0_crop,img1_n25,mu,sigma))

    println("5): \n Compute Laplacian negative Log Likelyhood for 0% noise")
    println("Laplacian nllh: ", laplacian_nllh(img0_crop,img1_d_crop,mu,s))
    println("Compute Laplacian negative Log Likelyhood for 12% noise")
    println("Laplacian nllh: ", laplacian_nllh(img0_crop,img1_n12,mu,s))
    println("Compute Laplacian negative Log Likelyhood for 25% noise")
    println("Laplacian nllh: ", laplacian_nllh(img0_crop,img1_n25,mu,s))

    println("6): \nThe Gausian negativ log Likelyhood for 12% noise rises by a factor of ",gaussian_nllh(img0_crop,img1_n12,mu,sigma)/gaussian_nllh(img0_crop,img1_d_crop,mu,sigma))
    println("The Laplacian negativ log Likelyhood for 12% noise rises by a factor of ",laplacian_nllh(img0_crop,img1_n12,mu,sigma)/laplacian_nllh(img0_crop,img1_d_crop,mu,sigma))

    println("The Gausian negativ log Likelyhood for 25% noise rises by a factor of ",gaussian_nllh(img0_crop,img1_n25,mu,sigma)/gaussian_nllh(img0_crop,img1_d_crop,mu,sigma))
    println("The Laplacian negativ log Likelyhood for 25% noise rises by a factor of ",laplacian_nllh(img0_crop,img1_n25,mu,sigma)/laplacian_nllh(img0_crop,img1_d_crop,mu,sigma))


end
