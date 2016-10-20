# Apply variable selection using power likelihood to the CPP (collaborative perinatal project data)
include("varsel.jl")

module RunCPP
using VarSel
using PyPlot
draw() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())

# Settings
n_total = 10^4  # total number of MCMC samples
n_keep = n_total  # number of MCMC samples to record
alphas = [100,500,1000,2000,Inf]  # values of alpha to use
nburn = int(n_keep/10)  # burn-in

# Variables to use
target_name = "V_BWGT"  # name of target variable
predictor_names = vec(readdlm("datasets/CPP/predictors.txt"))  # names of the predictor variables / covariates

# randomly permute order to check for possible MCMC mixing issues
#predictor_names = predictor_names[randperm(length(predictor_names))]

# Load data
data = readdlm("datasets/CPP/finaldde.csv",',')
names = vec(data[1,:])
values = data[2:end,:]
predictor_subset = [findfirst(names.==name)::Int64 for name in predictor_names]
@assert(minimum(predictor_subset)>0)
target_index = findfirst(names.==target_name)
for j in predictor_subset
    missing = vec(values[:,j].==".")
    values[missing,j] = mean(values[!missing,j])
end
println("Missing values in covariates have been replaced by the average of non-missing entries.")
missing_y = vec(values[:,target_index].==".")
y = convert(Array{Float64,1}, values[!missing_y,target_index])
x = convert(Array{Float64,2}, values[!missing_y,predictor_subset])
n = length(y)
println("Removed $(sum(missing_y)) records which were missing the target variable.")

# Preprocess data
y = (y-mean(y))./std(y)
x = (x.-mean(x,1))./std(x,1)
@assert(all(!isnan(y)))
@assert(all(!isnan(x)))
x = [ones(n) x] # append constant covariate for offset
predictor_names = ["constant",predictor_names]
p = size(x,2)

function interval(x,P)
    N = length(x)
    xsort = sort(x)
    ai = (1-P)/2
    l,u = xsort[int(floor(ai*N))], xsort[int(ceil((1-ai)*N))]
    return l,u
end


# Run sampler
nalphas = length(alphas)
use = nburn+1:n_keep
n_use = n_keep - nburn
k_posteriors = zeros(p+1,nalphas)
P_posteriors = zeros(p,nalphas)
beta_means = zeros(p,nalphas)
for (i_a,alpha) in enumerate(alphas)

    # Run sampler
    zeta = (1/n)/(1/n + 1/alpha)
    beta_r,lambda_r,keepers = VarSel.run(x,y,n_total,n_keep,zeta)
    k_r = vec(sum(beta_r.!=0, 1))
    P_nonzero = vec(mean(beta_r[:,use].!=0, 2))
    P_posteriors[:,i_a] = P_nonzero
    beta_means[:,i_a] = vec(mean(beta_r[:,use],2))

    # Compute posterior on k
    edges,counts = hist(k_r[use],-0.5:1:p+0.5)
    k_posteriors[:,i_a] = counts/n_use

    # Show traceplot of number of nonzero coefficients
    figure(1); clf(); hold(true)
    plot(k_r + 0.5*rand(length(k_r))-0.25,"k.",markersize=2)
    draw()
    savefig("cpp-traceplot-alpha=$alpha.png",dpi=150)
    
    # Show posterior of lambda
    figure(2); clf(); hold(true)
    plt.hist(lambda_r[use], 50)
    draw()
    savefig("cpp-lambda-alpha=$alpha.png",dpi=150)
    println("E(lambda|data) = ", mean(lambda_r[use]))

    # List of most probable nonzero coefs
    order = sortperm(P_nonzero,rev=true)

    # Show traceplots of top coefficients
    figure(3,figsize=(8,8)); clf(); hold(true)
    subplots_adjust(bottom=0.1,hspace=0.4)
    for ji = 1:8
        j = order[ji]
        subplot(8,1,ji)
        betas = vec(beta_r[j,:])
        plot(betas,"k.",markersize=2)
        xm,xM = xlim(); ym,yM = ylim()
        text((xM-xm)*0.05+xm,(yM-ym)*0.5+ym,predictor_names[j])
    end
    draw()
    savefig("cpp-traceplot-beta-alpha=$alpha.png",dpi=150)
    
    # Display c.d.f.s of coefficients
    figure(4,figsize=(8,8)); clf(); hold(true)
    subplots_adjust(bottom=0.1,hspace=0.4)
    for ji = 1:15
        j = order[ji]
        subplot(5,3,ji)
        betas = vec(beta_r[j,use])
        plot(sort([betas,betas]),sort([0:n_use-1,1:n_use])/n_use,"b-",markersize=0,linewidth=2)
        ylim(0,1)
        mb = max(0.01, maximum(abs(betas)))
        xlim(-mb,mb)
        xm,xM = xlim(); ym,yM = ylim()
        text((xM-xm)*0.05+xm,(yM-ym)*0.5+ym,predictor_names[j])
        l,u = interval(betas,0.95)
        plot([l,l],ylim(),"r-",markersize=0)
        plot([u,u],ylim(),"r-",markersize=0)
    end
    draw()
    savefig("cpp-beta-alpha=$alpha.png",dpi=150)

end

# Plot posteriors on k
figure(5,figsize=(8,3.2)); clf(); hold(true)
subplots_adjust(bottom=0.2)
colors = "bgyrm"
shapes = "ds^v*"
plot(0:p,VarSel.prior_k(0:p,p),"ko-",label="prior",markersize=8,linewidth=2)
for (i_a,alpha) in enumerate(alphas)
    lab = "\$\\alpha = " * (alpha<Inf ? repr(int(alpha)) : "\\infty") * "\$"
    plot(0:p,vec(k_posteriors[:,i_a]),"$(colors[i_a])$(shapes[i_a])-",label=lab,markersize=8,linewidth=2)
end
xticks(0:p)
xlim(0,16)
ylim(0,1)
xlabel("\$k \\,\\,\\mathrm{(\\#\\,of\\,nonzero\\,coefficients)}\$",fontsize=17)
ylabel("\$\\Pi(k|\\mathrm{data})\$",fontsize=15)
legend(numpoints=1)
draw()
savefig("cpp-k_posteriors.png",dpi=150)

# Sort coefs by overall probability of inclusion
order = sortperm(vec(mean(P_posteriors,2)),rev=true)
names_order = predictor_names[order]
for i in 1:16
    println(i,": ",names_order[i])
end

# Plot posterior probability of inclusion
figure(6,figsize=(8,3.2)); clf(); hold(true)
subplots_adjust(bottom=0.2)
colors = "bgyrm"
shapes = "ds^v*"
for (i_a,alpha) in enumerate(alphas)
    lab = "\$\\alpha = " * (alpha<Inf ? repr(int(alpha)) : "\\infty") * "\$"
    plot(1:p,vec(P_posteriors[order,i_a]),"$(colors[i_a])$(shapes[i_a])-",label=lab,markersize=8,linewidth=2)
end
d_show = 16
xticks(1:d_show)
xlim(0,d_show)
yticks(linspace(0,1,6))
ylim(0,1.1)
xlabel("\$j\\,\\,\\mathrm{(coefficient\\,index)}\$",fontsize=17)
ylabel("\$\\Pi(\\beta_j \\neq 0\\,|\\,\\mathrm{data})\$",fontsize=16)
legend(numpoints=1)
draw()
savefig("cpp-P_posteriors.png",dpi=150)


# Plot posterior means
figure(7,figsize=(8,3.2)); clf(); hold(true)
subplots_adjust(bottom=0.2)
colors = "bgyrm"
shapes = "ds^v*"
for (i_a,alpha) in enumerate(alphas)
    lab = "\$\\alpha = " * (alpha<Inf ? repr(int(alpha)) : "\\infty") * "\$"
    plot(1:p,vec(beta_means[order,i_a]),"$(colors[i_a])$(shapes[i_a])-",label=lab,markersize=8,linewidth=2)
end
d_show = 16
xticks(1:d_show)
xlim(0,d_show)
xlabel("\$j\\,\\,\\mathrm{(coefficient\\,index)}\$",fontsize=17)
ylabel("\$E(\\beta_j |\\mathrm{data})\$",fontsize=16)
legend(numpoints=1)
draw()
savefig("cpp-beta_means.png",dpi=150)


end # module


