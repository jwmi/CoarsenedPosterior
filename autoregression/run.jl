# Apply c-posterior to autoregressive model of variable order.
# Using analytical expression for evidence, assuming known variance.
module Run
using PyPlot
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())

# Settings
ns = [10 .^ [2:4]]
alphas = [Inf,500] # robustification param (max sample size)
K = 20 # maximum value of k to use

# Data generation parameters
s0 = 1.0 # std dev of noise
sm = 0.5 # scale of misspecification
a0 = [0.25,0.25,-0.25,0.25] # autoregression coefficients
k0 = length(a0)
f_noise(t) = randn()*s0 + sm*sin(t) # Time-varying noise

# Model parameters
s = s0
sa = 1.0 # std dev of prior on a's

# Code for autoregressive model
include("AR.jl")

srand(0) # Reset RNG
x_all = generate(maximum(ns),a0,f_noise) # Generate data
for (i_a,alpha) in enumerate(alphas)
    for (i_n,n) in enumerate(ns)
        zeta = (1/n)/(1/n + 1/alpha) # robustification param
        x = x_all[1:n]

        # Compute marginal likelihood for each k
        log_m = zeros(K+1)
        for k = 0:K
            log_m[k+1] = log_marginal(x,k,s,sa,zeta)
        end

        # Plot
        figure(i_a*10+i_n,figsize=(8,2.5)); clf(); hold(true)
        subplots_adjust(top=.8,bottom=0.25,left=.13)
        plot(0:K,log_m,"ko-",markersize=8,linewidth=2)
        xlim(0,K)
        xticks(0:K)
        mx,mn = maximum(log_m),minimum(log_m)
        pad = (mx-mn)/20
        ylim(mn-pad,mx+pad)
        ticklabel_format(axis="y",style="sci",scilimits=(-4,4))
        if alpha<Inf; title("\$\\mathrm{Coarsened},\\,n=$n\$",fontsize=17)
        else; title("\$\\mathrm{Standard},\\,n=$n\$",fontsize=17)
        end
        xlabel("\$k \\,\\,\\mathrm{(\\#\\,of\\,coefficients)}\$",fontsize=16)
        if n==ns[1]
            ylabel("\$\\log\\,p(\\mathrm{data}|k)\$",fontsize=15)
        end
        draw()
        savefig("AR-alpha=$alpha-n=$n.png",dpi=150)
    end
end

figure(3,figsize=(8,3.2)); clf(); hold(true)
subplots_adjust(bottom=0.2)
x = generate(200,a0,f_noise)
plot(x)
title("\$\\mathrm{AR($k0)\\,data\\,with\\,time-varying\\,noise}\$",fontsize=17)
xlabel("\$t \\, \\mathrm{(time)}\$",fontsize=16)
ylabel("\$x_t\$",fontsize=18)
ylim(-4,4)
draw()
savefig("AR-data.png",dpi=150)

end # module


