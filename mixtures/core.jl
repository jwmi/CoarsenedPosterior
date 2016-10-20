# Map-MCMC for mixtures.

# This code assumes the following functions have been defined:
#   likelihood(x[j],theta)
#   log_v_prior(v)
#   log_theta_prior(theta)
#   theta_prop(theta)
#   log_theta_prop(theta,thetap)
#   new_thetas(m)

function sampler(x,n_samples,m,c,sigma,zeta)
    # x = data (array of datapoints)
    # n_samples = # of MCMC iterations
    # m = maximum # of mixture components
    # c = cutoff point for weight map
    # sigma = scale of weight proposals
    # zeta = power to raise likelihood to

    n = length(x)
    
    # initialize state
    v = zeros(m); v[1] = 2*c; v[2:end] = c/2 # latent weights
    q = max(v-c,0) # mapped weights (unnormalized)
    s = sum(q)
    theta = new_thetas(m)

    # initialize vars for computing likelihood
    L = [likelihood(x[j],theta[i]) for i=1:m, j=1:n]
    M = vec(q'*L)  # mixture density with unnormalized weights
    ll = sum(log(M)) - n*log(s)  # log-lik
    Mp = zeros(n)  # Mp and Lp will hold proposed values
    Lp = zeros(n)

    # record-keeping
    k_r = zeros(n_samples)
    m_r = zeros(n_samples,m)
    l_r = zeros(n_samples,m)
    v_r = zeros(n_samples,m)
    nta = 0  # number of theta proposals accepted
    nva = 0  # number of v proposals accepted
    
    # draw samples
    for iter = 1:n_samples
        # update parameters with Metropolis-Hastings moves
        for i = 1:m
            thetap = theta_prop(theta[i])
            llp = -n*log(s)
            for j = 1:n
                Lp[j] = likelihood(x[j],thetap)
                Mp[j] = max(M[j] + q[i]*(Lp[j] - L[i,j]), 0)  
                # Note: max(.,0) prevents negative values due to roundoff error.
                llp += log(Mp[j])
            end
            # compute acceptance probability
            top = log_theta_prior(thetap) + zeta*llp + log_theta_prop(thetap,theta[i])
            bot = log_theta_prior(theta[i]) + zeta*ll + log_theta_prop(theta[i],thetap)
            p_accept = min(1, exp(top-bot))
            # accept or reject
            if rand() < p_accept
                theta[i] = copy(thetap)
                for j = 1:n; L[i,j] = Lp[j]; end
                M,Mp = Mp,M
                ll = llp
                nta += 1
            end
        end
        
        # update weights with Metropolis-Hastings moves
        for i = 1:m
            vp = v[i]*exp(randn()*sigma)
            qp = max(vp-c,0)
            sp = s + qp - q[i]
            if sp > 0
                llp = -n*log(sp)
                for j = 1:n
                    Mp[j] = max(M[j] + (qp - q[i])*L[i,j], 0)
                    llp += log(Mp[j])
                end
                # compute acceptance probability
                top = log_v_prior(vp) + zeta*llp - log(v[i])
                bot = log_v_prior(v[i]) + zeta*ll - log(vp)
                p_accept = min(1, exp(top-bot))
                # accept or reject
                if rand() < p_accept
                    v[i] = vp
                    q[i] = qp
                    s = sp
                    M,Mp = Mp,M
                    ll = llp
                    nva += 1
                end
            end
        end

        # record
        k_r[iter] = sum(q.>0)  # number of active components
        for i = 1:m
            m_r[iter,i] = theta[i][1]  # means
            l_r[iter,i] = theta[i][2]  # precisions
            v_r[iter,i] = v[i]  # latent weights
        end

        # visualize mixture components during sampling
        if false
            figure(10); clf(); hold(true)
            xs = linspace(-5,5,1000)
            for i = 1:m
                if q[i]>0
                    plot(xs,(q[i]/s)*normpdf(xs,theta[i][1],theta[i][2]))
                end
            end
            draw()
        end
    end
    art = nta/(m*n_samples)  # MH acceptance rate for theta proposals
    arv = nva/(m*n_samples)  # MH acceptance rate for v proposals
    return (q/s),theta,k_r,v_r,art,arv,m_r,l_r
end





