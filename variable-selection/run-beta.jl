# Posterior c.d.f.s for simulation example for variable selection using power likelihood
include("varsel.jl")

module RunBeta
using VarSel
using PyPlot
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())

# Settings
ns = [10000]  # sample sizes to use
alphas = [Inf,50]  # values of alpha to use
nreps = 1  # number of times to repeat each simulation
n_total = 5*10^4  # total number of MCMC samples
n_keep = n_total  # number of MCMC samples to record
nburn = int(n_keep/10)  # burn-in

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

function interval(x,P)
    # Compute 100*P% interval from samples x
    N = length(x)
    xsort = sort(x)
    ai = (1-P)/2
    l,u = xsort[int(floor(ai*N))], xsort[int(ceil((1-ai)*N))]
    return l,u
end

# Run simulations
nns = length(ns)
for (i_a,alpha) in enumerate(alphas)
    for (i_n,n) in enumerate(ns)
        for rep in 1:nreps
            srand(n+rep) # Reset RNG
            
            # Sample data
            x,y = generate_sample(n)
            x0 = vec(x[:,2])

            # Run sampler
            zeta = (1/n)/(1/n + 1/alpha)
            beta_r,lambda_r,keepers = VarSel.run(x,y,n_total,n_keep,zeta)
                
            # Display c.d.f.s of coefficients
            xlims = Array[[-1.5,0.1],[2,4.5],[-.4,.4],[-.4,.4],[-.4,.4],[-.4,.4]]
            figure(i_a+2,figsize=(8,8)); clf(); hold(true)
            subplots_adjust(bottom=0.1,hspace=0.4)
            for j=1:p
                subplot(p,1,j)
                betas = vec(beta_r[j,nburn+1:end])
                n_use = n_keep - nburn
                plot(sort([betas,betas]),sort([0:n_use-1,1:n_use])/n_use,"b-",markersize=0,linewidth=2)
                ylim(0,1)
                xlim(xlims[j])
                xm,xM = xlim(); ym,yM = ylim()
                text((xM-xm)*0.92+xm,(yM-ym)*0.5+ym,"\$\\beta_$j\$",fontsize=18)
                l,u = interval(betas,0.95)
                plot([l,l],ylim(),"r-",markersize=0)
                plot([u,u],ylim(),"r-",markersize=0)
            end
            subplot(p,1,1)
            ylabel("\$\\\mathrm{c.d.f.}\$",fontsize=16)
            if alpha<Inf; title("\$\\mathrm{Coarsened}\$",fontsize=17)
            else; title("\$\\mathrm{Standard}\$",fontsize=17)
            end
            draw()
            savefig("varsel-beta-alpha=$alpha-n=$n-rep=$rep.png",dpi=150)
        end
    end
end


end # module


