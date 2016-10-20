# Sample densities from the c-posterior, for the skew-normal example
module Densities
using Distributions
using PyPlot
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())


# Settings
n = 10000  # sample size to use
alphas = [Inf,100]  # robustification params to use
n_samples = 10^4  # number of MCMC iterations

# Model parameters and sampler code
include("setup.jl")

for (i_a,alpha) in enumerate(alphas)
    x = [(rand()<p1 ? skewrnd(1,l1,s1,a1) : skewrnd(1,l2,s2,a2))[1]::Float64 for j=1:n] # Sample data

    # Run sampler
    zeta = (1/n)/(1/n + 1/alpha)
    p,theta,k_r,v_r,art,arv,m_r,s_r = sampler(x,n_samples,m,c,sigma,zeta)

    # Plot
    figure(i_a,figsize=(8,3.2)); clf(); hold(true)
    subplots_adjust(bottom=0.2)
    xmin,xmax = -5,5
    xs = linspace(xmin,xmax,1000)
    density = zeros(length(xs))
    for i = 1:m
        if p[i]>0
            pixs = p[i]*normpdf(xs,theta[i][1],theta[i][2])
            density += pixs
            plot(xs,pixs)
        end
    end
    plot(xs,density,"k--",linewidth=3)
    xlim(xmin,xmax)
    ylim(0,0.4)
    if alpha<Inf; title("\$\\mathrm{Coarsened\\,posterior}\$",fontsize=17)
    else; title("\$\\mathrm{Standard\\,posterior}\$",fontsize=17)
    end
    xlabel("\$x\$",fontsize=16)
    ylabel("\$\\mathrm{density}\$",fontsize=15)
    draw()
    savefig("density-alpha=$alpha.png",dpi=150)
end


end # module



