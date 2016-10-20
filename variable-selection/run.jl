# Simulation example for variable selection using power likelihood
include("varsel.jl")

module Run
using Distributions, HDF5, JLD
using VarSel
using PyPlot
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())

# Settings
ns = [100,1000,5000,10000,50000]  # sample sizes to use
alphas = [Inf,50]  # values of alpha to use
nreps = 10  # number of times to repeat each simulation
n_total = 5*10^4  # total number of MCMC samples
n_keep = 10^4  # number of MCMC samples to record
nburn = int(n_keep/10)  # burn-in
from_file = false  # load previous results from file

# Data distribution
using Skew
a = [0.6,2.7,-3.3,-4.9,-2.5]
Q = [1.0 -0.89 0.93 -0.91 0.98
     -0.89 1.0 -0.94 0.97 -0.91
     0.93 -0.94 1.0 -0.96 0.97
     -0.91 0.97 -0.96 1.0 -0.93
     0.98 -0.91 0.97 -0.93 1.0]
sigma = 1
p = length(a)+1

g(x0) = -1 + 4*(x0 + (1/16)*x0.^2)

function generate_sample(n)
    x = [ones(n) Skew.skewrndNormalized(n,Q,a)']
    x0 = vec(x[:,2])
    y = g(x0) + randn(n)*sigma
    return x,y
end

# Plot data
srand(1)
figure(5,figsize=(8,3.2)); clf(); hold(true)
subplots_adjust(bottom=0.2)
n = 200
xmin,xmax = -4,4
ymin,ymax = -15,15
xs,ys = generate_sample(n)
xs0 = vec(xs[:,2])
plot(xs0,ys,"b.",markersize=3)
xt = linspace(xmin,xmax,5000)
plot(xt,g(xt),"k-")
xlim(xmin,xmax); ylim(ymin,ymax)
title("\$\\mathrm{Data\\,distribution}\$",fontsize=17)
xlabel("\$x_{i 2}\$",fontsize=18)
ylabel("\$y_i\$",fontsize=18)
draw()
savefig("varsel-data-n=$n.png",dpi=150)

# Run simulations
nns = length(ns)
for (i_a,alpha) in enumerate(alphas)
    if from_file
        k_posteriors = load("k_posteriors-alpha=$alpha.jld","k_posteriors")
    else
        k_posteriors = zeros(p+1,nreps,nns)
        for (i_n,n) in enumerate(ns)
            for rep in 1:nreps
                srand(n+rep) # Reset RNG
                
                # Sample data
                x,y = generate_sample(n)
                x0 = vec(x[:,2])

                # Run sampler
                zeta = (1/n)/(1/n + 1/alpha)
                beta_r,lambda_r,keepers = VarSel.run(x,y,n_total,n_keep,zeta)
                k_r = vec(sum(beta_r.!=0, 1))

                # Compute posterior on k
                edges,counts = hist(k_r[nburn+1:end],-0.5:1:p+0.5)
                k_posteriors[:,rep,i_n] = counts/(n_keep-nburn)

                # traceplot of number of nonzero coefficients
                figure(10); clf(); hold(true)
                plot(k_r + 0.5*rand(length(k_r))-0.25,"k.",markersize=2)
                draw()
                #savefig("traceplot-alpha=$alpha-n=$n-rep=$rep.png",dpi=150)
            end
        end
        save("k_posteriors-alpha=$alpha.jld","k_posteriors",k_posteriors)
    end
    
    # Plot results
    figure(i_a,figsize=(8,3.2)); clf(); hold(true)
    subplots_adjust(bottom=0.2,right=0.75)
    colors = "bgyrm"
    shapes = "ds^v*"
    plot(0:p,VarSel.prior_k(0:p,p),"ko-",label="prior",markersize=8,linewidth=2)
    for (i_n,n) in enumerate(ns)
        plot(0:p,vec(mean(k_posteriors[:,:,i_n],2)),"$(colors[i_n])$(shapes[i_n])-",label="n = $n",markersize=8,linewidth=2)
    end
    xlim(0,p)
    xticks(0:p)
    if alpha<Inf; title("\$\\mathrm{Coarsened}\$",fontsize=17)
    else; title("\$\\mathrm{Standard}\$",fontsize=17)
    end
    xlabel("\$k \\,\\,\\mathrm{(\\#\\,of\\,nonzero\\,coefficients)}\$",fontsize=17)
    ylabel("\$\\Pi(k|\\mathrm{data})\$",fontsize=15)
    legend(numpoints=1,bbox_to_anchor=(1.02, 0.9), loc=2, borderaxespad=0.)
    draw()
    savefig("varsel-alpha=$alpha.png",dpi=150)
end

end # module


