# Skew-normal simulation example for c-posterior for mixtures
module Suite
using Distributions
using PyPlot
using HDF5, JLD
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())

# Settings
ns = [20,100,500,2000,10000]  # sample sizes n to use
alphas = [Inf,100]  # robustification params alpha to use
nreps = 5  # number of times to run the simulation
n_samples = 10^5  # number of MCMC iterations
nburn = int(n_samples/10)  # number of iterations to discard as burn-in
from_file = false  # load previous results from file

# Model parameters and sampler code
include("setup.jl")

nns = length(ns)
for (i_a,alpha) in enumerate(alphas)

    if from_file
        k_posteriors = load("k_posteriors-alpha=$alpha.jld","k_posteriors")
    else
        k_posteriors = zeros(m,nreps,nns)
        for (i_n,n) in enumerate(ns)
            for rep in 1:nreps
                srand(n+rep) # Reset RNG
                x = [(rand()<p1 ? skewrnd(1,l1,s1,a1) : skewrnd(1,l2,s2,a2))[1]::Float64 for j=1:n] # Sample data

                # Run sampler
                zeta = (1/n)/(1/n + 1/alpha)
                p,theta,k_r,v_r,art,arv,m_r,l_r = sampler(x,n_samples,m,c,sigma,zeta)

                # Compute posterior on k
                edges,counts = hist(k_r[nburn+1:end],[0:m])
                k_posteriors[:,rep,i_n] = counts/(n_samples-nburn)
            end
        end
        save("k_posteriors-alpha=$alpha.jld","k_posteriors",k_posteriors)
    end

    # Prior on k: p(k) \propto Binomial(k|m,lambda/m)I(k>0).
    pk = pdf(Binomial(m,lambda/m),1:m); pk = pk/sum(pk)

    # Plot
    figure(i_a,figsize=(8,3.2)); clf(); hold(true)
    subplots_adjust(bottom=0.2)
    colors = "bgyrm"
    shapes = "ds^v*"
    plot(1:m,pk,"ko-",label="prior",markersize=8,linewidth=2)
    for (i_n,n) in enumerate(ns)
        plot(1:m,vec(mean(k_posteriors[:,:,i_n],2)),"$(colors[i_n])$(shapes[i_n])-",label="n = $n",markersize=8,linewidth=2)
    end
    xlim(0,m)
    xticks(1:m)
    if alpha<Inf; title("\$\\mathrm{Coarsened\\,posterior}\$",fontsize=17)
    else; title("\$\\mathrm{Standard\\,posterior}\$",fontsize=17)
    end
    xlabel("\$k \\,\\,\\mathrm{(\\#\\,of\\,components)}\$",fontsize=17)
    ylabel("\$\\Pi(k|\\mathrm{data})\$",fontsize=15)
    legend(loc="upper right",numpoints=1)
    draw()
    savefig("skewsuite-alpha=$alpha.png",dpi=150)
    
end

end # module



