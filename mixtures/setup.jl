# Supporting parameters and functions for skew-normal simulation example

# Stuff for mixture weights
m = 10 # maximum number of components
a = 1/m # Gamma parameters
G = Gamma(a,1/1) # prior on weights (before conditioning on s>0)
lambda = 1.0  # Poisson parameter of the limiting prior distribution on k as m -> Inf
c = quantile(G,1-lambda/m)  # cutoff (Note: This choice makes p(k) \propto Binomial(k|m,lambda/m)I(k>0).)
sigma = 0.25 # scale for weight proposals
log_v_prior(v_i) = logpdf(G,v_i) # prior on latent weights v_i


# Stuff for mixture components
normpdf(x,m,l) = sqrt(l/(2*pi))*exp(-0.5*l*(x-m).*(x-m))
normlogpdf(x,m,s) = -0.5*log(2*pi) - log(s) - 0.5*(x-m)*(x-m)/(s*s)
likelihood(x,t) = normpdf(x,t[1],t[2])  # t=[mean,precision]
# prior (base measure) on component params, [mean,log(precision)]
m0m,s0m = 0,5
m0l,s0l = 0,2
log_theta_prior(t) = normlogpdf(t[1],m0m,s0m) + normlogpdf(log(t[2]),m0l,s0l)
# proposal distribution for component param moves
sm,sl = 0.2*s0m, 0.2
theta_prop(t) = Float64[t[1]+sm*randn(), t[2]*exp(randn()*sl)]
log_theta_prop(t,tp) = normlogpdf(tp[1],t[1],sm) + normlogpdf(log(tp[2]),log(t[2]),sl)
new_thetas(m) = [Float64[m0m,exp(m0l)] for i=1:m]  # initialize params for m new components


# Densities, etc. for skew-normal
normpdf(x) = exp(-0.5*x.*x)/sqrt(2.0*pi)
normcdf(x) = 0.5*(1.0+erf(x/sqrt(2.0)))
skewpdf(x,loc,scale,shape) = (y=(x-loc)/scale; 2.0*normpdf(y).*normcdf(shape*y)/scale)
skewrnd(n,loc,scale,shape) = [(shape*z<randn() ? loc-scale*z : loc+scale*z) for z=randn(n)]


# Params of skew-normal mixture to use in simulation
p1 = 0.5  # probability of component 1
l1,s1,a1 = -4,1,5  # location, scale, and shape of component 1
l2,s2,a2 = -1,2,5  # location, scale, and shape of component 2


# Sampler code
include("core.jl")


