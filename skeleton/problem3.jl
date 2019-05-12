# Problem 3: Getting to know Julia
#fix for PyPlot qt runtime error
#ENV["MPLBACKEND"]="qt4agg"
#ENV["MPLBACKEND"]="tkagg"
# here you go..
using PyPlot
using Images
using Colors
using Statistics


function main()
  # Use PyPlot to load image
  img = PyPlot.imread("a1p3.png")
  println(typeof(img))
  #convert to grayscale and fortunatly this converts the Array to Float64
  imggg = 0.2989 * img[:,:,1] + 0.5870 * img[:,:,2] + 0.1140 * img[:,:,3];
  println(typeof(imggg));
  #calc max,min,mean
  println("Max: ",maximum(imggg)," Min: ",minimum(imggg)," Mean: ",mean(imggg));
  #Plot with PyPlot
  PyPlot.axis("off");
  PyPlot.imshow(imggg,cmap="gray");
  PyPlot.title("Problem3 Gray scaled");
  fig1 = PyPlot.gcf();
  display(fig1);
end
