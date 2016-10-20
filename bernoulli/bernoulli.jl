# Code for toy example involving Bernoulli trials
module bernoulli_example
using PyPlot, Distributions
draw_now() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
logsumexp(x) = (m = maximum(x); m == -Inf ? -Inf : log(sum(exp(x-m))) + m)

# H0: theta = 1/2
# H1: theta \neq 1/2, theta|H1 ~ Beta(1,1)

# settings
epsilon = 0.02 # "precision" of theta
alpha = 1/(2*epsilon^2) # choose alpha using relative entropy approach, for theta near 1/2
theta0 = 0.56 # true value
nns = 25 # 1+6*4 = number of n's to use
nreps = 1000 # number of times to run the simulation
ns = int(logspace(0,6,nns)) # n's to use

# initialize
p0xs = zeros(nreps,nns)
r0xs = zeros(nreps,nns)
e0xs = zeros(nreps,nns)
maxn = maximum(ns)

for rep = 1:nreps
    y = (rand(maxn).<theta0) # data
    for (i,n) in enumerate(ns)
        s = sum(y[1:n])

        # standard posterior
        B10 = exp(n*log(2) + lbeta(1+s, 1+n-s)) # p(x|H1)/p(x|H0)
        p0x = 1/(1+B10)  # p(H0|x)
        
        # robust posterior - approximate
        alpha_n = 1/(1/alpha + 1/n)
        R10 = exp(alpha_n*log(2) + lbeta(1+alpha_n*s/n, 1+alpha_n*(1-s/n)))
        r0x = 1/(1+R10)  # p(H0|E)

        # robust posterior - exact
        f(p,q) = (r=p.*log(p./q); if length(p)>1; r[p.==0]=0; elseif p==0; r[:]=0; end; r)
        D(p,q) = f(p,q) + f(1-p,1-q) # relative entropy for Bernoulli
        t = [0:n]
        lpt0 = logpdf(Binomial(n,1/2),t) # Binomial(n,1/2)
        lpt1 = -log(n+1)*ones(n+1) # BetaBinomial(n,1,1) = Uniform{0,1,...,n}
        lpE0 = logsumexp(-alpha*D(s/n,t/n) + lpt0)  # log(p(E|H0))
        lpE1 = logsumexp(-alpha*D(s/n,t/n) + lpt1)  # log(p(E|H1))
        E10 = exp(lpE1 - lpE0)
        e0x = 1/(1+E10)  # p(H0|E)

        # record values
        p0xs[rep,i] = p0x
        r0xs[rep,i] = r0x
        e0xs[rep,i] = e0x

        #println("n = $n")
        #@printf("p(H0|x)  std=%.4f  approx=%.5f  exact=%.5f\n",p0x,r0x,e0x)
    end
end

# plot
figure(1,figsize=(10,3.2)); clf(); hold(true); grid(true)
subplots_adjust(bottom = 0.2)
semilogx(ns,mean(p0xs,1)[:],label="standard posterior",linewidth=2,"gs-",markersize=5)
semilogx(ns,mean(e0xs,1)[:],label="exact c-posterior",linewidth=2,"r*-",markersize=10)
semilogx(ns,mean(r0xs,1)[:],label="approx c-posterior",linewidth=1,"bo-",markersize=5)
xlabel("\$n \\,\\,\\mathrm{(sample\\,size)}\$",fontsize=16)
ylabel("\$\\Pi(\$H\$_0|\\mathrm{data})\$",fontsize=15)
ylim(0,1)
#legend(numpoints=1,loc="lower left",fontsize=14)
draw_now()
savefig("pH0-$theta0.png",dpi=150)

end

