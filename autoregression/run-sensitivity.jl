# Sensitivity analysis for c-posterior of autoregressive model of variable order.
module Run
using PyPlot
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
logsumexp(x) = (m=maximum(x); (m==-Inf ? -Inf : log(sum(exp(x-m)))+m))

# Settings
n = 10^4
alphas = [50,100,500,1200,2000] # robustification param (max sample size)
K = 25 # maximum value of k to use

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

figure(1,figsize=(8,3.2)); clf(); hold(true)
subplots_adjust(bottom=0.2)

srand(0) # Reset RNG
x = generate(n,a0,f_noise) # Generate data
for (i_a,alpha) in enumerate(alphas)
    zeta = (1/n)/(1/n + 1/alpha) # robustification param

    # Compute marginal likelihood for each k
    log_m = zeros(K+1)
    for k = 0:K
        log_m[k+1] = log_marginal(x,k,s,sa,zeta)
    end

    # Compute posterior on k, using an improper uniform prior on k
    pk = exp(log_m - logsumexp(log_m))

    # Plot
    colors = "bgyrm"
    shapes = "ds^v*"
    lab = "\$\\alpha = " * (alpha<Inf ? repr(int(alpha)) : "\\infty") * "\$"
    plot(0:K,pk,"$(colors[i_a])$(shapes[i_a])-",label=lab,markersize=8,linewidth=2)
end


title("\$\\mathrm{Sensitivity\\,analysis}\$",fontsize=17)
ylabel("\$\\Pi(k|\\mathrm{data})\$",fontsize=15)
xlabel("\$k \\,\\,\\mathrm{(\\#\\,of\\,coefficients)}\$",fontsize=16)
xlim(0,12)
xticks(0:12)
ylim(0,1)
legend(numpoints=1)
draw()
savefig("AR-sensitivity.png",dpi=150)

end # module


