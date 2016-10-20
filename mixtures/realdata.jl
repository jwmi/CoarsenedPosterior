# Run mixture c-posterior on real-data examples
module RealData
using Distributions
using PyPlot
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())

# Settings
alphas = [20,100,500,1000,Inf]  # robustification params to use
Ns = [10^5,10^5,10^5,10^5,10^6]  # number of MCMC iterations for each alpha
nburns = [10^4,10^4,10^4,10^4,2.5*10^5]  # number of iterations to discard as burn-in for each alpha
data_ID = "Shapley"  # which dataset to use
m = 20 # maximum number of components

# Load data
if data_ID=="birth"
    dataset = readdlm("datasets/gestday.dat",' ')
    xmin,xmax = 180,380
elseif data_ID=="Shapley"
    dataset = readdlm("datasets/Shapley_galaxy.dat",' ')
    dataset = convert(Array{Float64,1}, dataset[:,4])/1000
    xmin,xmax = -5,50
elseif data_ID=="height"
    #D = readdlm("datasets/height_DEFG.dat",' ')
    D = readdlm("datasets/height_FG.dat",' ')
    subset = (18 .<= D[:,1])  # ages 18+
    dataset = vec(D[subset, 2])
    dataset = 0.393701*dataset  # convert from centimeters to inches
    xmin,xmax = minimum(dataset),maximum(dataset)
end
x = convert(Array{Float64,1}, dataset[:])
n = length(x)

# Stuff for mixture weights
a = 1/m # Gamma parameters
G = Gamma(a,1/1) # prior on weights (before conditioning on s>0)
lambda = 1.0  # Poisson parameter of the limiting prior distribution on k as m -> Inf
c = quantile(G,1-lambda/m)  # cutoff (Note: This choice makes p(k) \propto Binomial(k|m,lambda/m)I(k>0).)
sigma = 0.25 # scale for weight proposals
log_v_prior(v_i) = logpdf(G,v_i) # prior on latent weights v_i

# Stuff for mixture components
m0m,s0m = mean(x),std(x)  # data-dependent choices here
m0l = log(4/var(x))  # expect stddev of components to be around 1/2 the stddev of the overall data
s0l = 2.0

# Inference stuff
normpdf(x,m,l) = sqrt(l/(2*pi))*exp(-0.5*l*(x-m).*(x-m))
normlogpdf(x,m,s) = -0.5*log(2*pi) - log(s) - 0.5*(x-m)*(x-m)/(s*s)
likelihood(x,t) = normpdf(x,t[1],t[2])  # t=[mean,precision]
log_theta_prior(t) = normlogpdf(t[1],m0m,s0m) + normlogpdf(log(t[2]),m0l,s0l)
sm,sl = 0.2*s0m, 0.2
theta_prop(t) = Float64[t[1]+sm*randn(), t[2]*exp(randn()*sl)]
log_theta_prop(t,tp) = normlogpdf(tp[1],t[1],sm) + normlogpdf(log(tp[2]),log(t[2]),sl)
new_thetas(m) = [Float64[m0m,exp(m0l)] for i=1:m]

# Sampler code
include("core.jl")

# Run sampler
n_alphas = length(alphas)
k_posteriors = zeros(m,n_alphas)
for (i_a,alpha) in enumerate(alphas)

    # Run sampler
    n_samples = Ns[i_a]
    nburn = nburns[i_a]
    zeta = (1/n)/(1/n + 1/alpha)
    p,theta,k_r,v_r,art,arv,m_r,s_r = sampler(x,n_samples,m,c,sigma,zeta)
    
    # Compute posterior on k
    edges,counts = hist(k_r[nburn+1:end],[0:m])
    k_posteriors[:,i_a] = counts/(n_samples-nburn)
    
    # Plot traceplot of k
    figure(10,figsize=(8,3.2)); clf(); hold(true)
    plot(k_r,"k.",markersize=2)
    savefig("real-traceplot-alpha=$alpha.png",dpi=150)
    #draw()
    
    # Plot typical sample density
    figure(i_a,figsize=(8,3.2)); clf(); hold(true)
    subplots_adjust(bottom=0.2)
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
    if alpha<Inf; title("\$\\alpha = $(repr(int(alpha)))\$",fontsize=17)
    else; title("\$\\alpha = \\infty\$",fontsize=17)
    end
    xlabel("\$x\$",fontsize=16)
    ylabel("\$\\mathrm{density}\$",fontsize=15)
    #draw()
    savefig("real-density-alpha=$alpha.png",dpi=150)
end

# Plot posterior on k
figure(11,figsize=(8,3.2)); clf(); hold(true)
subplots_adjust(bottom=0.2)
colors = "bgyrm"
shapes = "ds^v*"
for (i_a,alpha) in enumerate(alphas)
    lab = "\$\\alpha = " * (alpha<Inf ? "$(repr(int(alpha)))\$" : "\\infty\$")
    plot(1:m,vec(k_posteriors[:,i_a]),"$(colors[i_a])$(shapes[i_a])-",label=lab,markersize=8,linewidth=2)
end
xlim(0,m)
xticks(1:m)
xlabel("\$k \\,\\,\\mathrm{(\\#\\,of\\,components)}\$",fontsize=17)
ylabel("\$\\Pi(k|\\mathrm{data})\$",fontsize=15)
legend(loc="upper right",numpoints=1)
draw()
savefig("real-k-posteriors.png",dpi=150)


# Plot data distribution
figure(12,figsize=(8,3.2)); clf(); hold(true)
subplots_adjust(bottom=0.2)
edges,counts = hist(x,linspace(xmin,xmax,81))
dx = edges[2]-edges[1]
bar(edges[1:end-1],(counts/n)/dx,dx,color="w")
xlim(xmin,xmax)
title("\$\\mathrm{Data\\,distribution}\$",fontsize=17)
xlabel("\$x\$",fontsize=16)
ylabel("\$\\mathrm{density}\$",fontsize=15)
draw()
savefig("real-data.png",dpi=150)


end # module



