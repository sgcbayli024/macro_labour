using LinearAlgebra, QuantEcon, Random, Makie, CairoMakie, Distributions

pars = (;β = 0.996, # discount factor
        ρ = 0.98, # autocorrelation of productivity process
        σe = 0.01, # standard deviation of productivity process
        nz = 15, # number of points in the productivity grid
        γ = 0.6, # coefficient of job finding rate
        κ = 1.89, # vacancy posting cost
        δ = 0.012, # separation rate
        σ = 0, # CRRA parameter
        α = 1, # search cost parameter
        χ = 2, # search cost exponent
        a = 1/3, # λ(s) = s^a job posting opportunities
        w_lb = .1, # lower bound of wage grid
        w_ub = 1, # upper bound of wage grid
        Δw = 0.025, # wage grid step
        b = 0.54, # benefit parameter
        b_end = 0.1, # prob of moving to lowest benefits
        s_lb = 0.2, # lower bound of search cost grid
        s_ub = 0.4, # upper bound of search cost grid
        Δs = 0.02, # search cost grid step
        toler = 1e-6, # tolerance for convergence
        maxiter = 100000, # maximum number of iterations
        Np = 40000, # number of agents
        T = 200, # number of periods
        Tb = 100 # burn-in periods
)

function income_process(p)
    (;ρ, σe, nz) = p
    mc = rouwenhorst(nz, ρ, σe)
    z_grid = exp.(mc.state_values)
    Π = mc.p
    return z_grid, Π 
end

function wage_grid(pars)
    (;w_lb, w_ub, Δw) = pars
    w_grid = range(w_lb, w_ub, step = Δw)
    return w_grid
end

function search_cost_grid(pars)
    (;s_lb, s_ub, Δs) = pars
    s_grid = range(s_lb, s_ub, step = Δs)
    return s_grid
end

function job_finding_rate(θ, pars)
    (;γ) = pars
    p = θ*(1. + θ^γ)^(-1/γ)
    return p
end

function job_opp(s, pars)
    (;a) = pars
    job_opp = s^a
    return job_opp
end

function benefits_grid(pars)
    (;b) = pars
    temp_w_grid = wage_grid(pars)
    benefits = b .* temp_w_grid
    return benefits
end

function utility(c,pars)
    (;σ) = pars
    
    if σ == 1
        u = log
    else
        u = x -> (x^(1-σ) - 1)/(1-σ)
    end

    return u(c)
end

function job_dest(z,w,pars)
    (;δ) = pars
    if z >= w
        return δ
    else
        return 1
    end
end

function search_costs(s,pars)
    (;α, χ) = pars
    return α * (s^χ)
end

function initial_J(pars)
    (;nz,w_ub,w_lb,Δw) = pars
    nw = (w_ub - w_lb)/Δw + 1
    nw = round(Int,nw)
    J = zeros(nw,nz)
    return J
end

function iterate_J(pars)
    (;nz, w_ub, w_lb, Δw, a, b, γ, κ, δ, α, χ, β, toler, maxiter) = pars
    nw = (w_ub - w_lb)/Δw + 1
    nw = round(Int,nw)
    J_init = initial_J(pars)
    J_new = copy(J_init)
    error = 1
    iter = 1
    z_grid, Π = income_process(pars)
    w_grid = wage_grid(pars)
    EV = zeros(nw,nz)
    while iter <= maxiter
        J_new = copy(J_init)
        for i in 1:nw
            for j in 1:nz
                EV[i,j] = 0
                for jp in 1:nz
                    EV[i,j] += J_init[i,jp] * Π[j,jp]
                end
            end
            J_init[i,:] = z_grid .- w_grid[i] .+ (β .* (1 .- job_dest.(z_grid,w_grid[i],Ref(pars)))) .* EV[i,:]
        end
        error = maximum(abs.(J_new - J_init))
        J_new = copy(J_init)
        if error < toler
            println("--------------------")
            println("Converged in $iter iterations")
            println("Error: $error")
            println("--------------------")
            break
        end
        if iter == maxiter
            println("Maximum number of iterations reached")
        end
        if iter == 1
            println("--------------------")
            println("Iteration: $iter")
            println("Error: $error")
            println("--------------------")
        end
        if iter % 100 == 0
            println("--------------------")
            println("Iteration: $iter")
            println("Error: $error")
            println("--------------------")
        end
        iter += 1
    end
    return J_new
end

function θ_probs(Θ,pars)
    (;γ, nz, w_lb,w_ub,Δw) = pars
    nw = (w_ub - w_lb)/Δw + 1
    nw = round(Int,nw)
    P = zeros(nw,nz)
    P = Θ .* (1 .+ Θ.^γ).^(-1/γ)
    return P
end

function initial_W(pars)
    (;nz,w_ub,w_lb,Δw) = pars
    nw = (w_ub - w_lb)/Δw + 1
    nw = round(Int,nw)
    W = ones(nw,nz)
    return W
end

function initial_U(pars)
    (;nz,w_ub,w_lb,Δw) = pars
    nw = (w_ub - w_lb)/Δw + 1
    nw = round(Int,nw)
    U = ones(nw,nz)
    return U
end

function iterate_W_U(pars)
    (;nz, w_ub, w_lb, Δw, s_lb, s_ub, Δs, a, b, γ, κ, δ, α, χ, β, toler, maxiter) = pars
    nw = (w_ub - w_lb)/Δw + 1
    nw = round(Int,nw)
    ns = (s_ub - s_lb)/Δs + 1
    ns = round(Int,ns)
    W_init = initial_W(pars)
    U_init = initial_U(pars)
    W_new = copy(W_init)
    U_new = copy(U_init)
    search_policy = zeros(nw,nz)
    posting_decision = zeros(nw, nz)
    w_hat = zeros(nw,nz)
    EW = zeros(nw,nz)
    EU = zeros(nw,nz,ns)
    final_util = zeros(ns)
    z_grid, Π = income_process(pars)
    w_grid = wage_grid(pars)
    b_grid = benefits_grid(pars)
    s_grid = search_cost_grid(pars)
    Θ = thetas(J_out,pars)
    P = θ_probs(Θ,pars)
    error = toler+1
    iter = 1
    while iter <= maxiter
        ### Value of employment ###
        for i in 1:nw
            for j in 1:nz
                EW[i,j] = 0
                for jp in 1:nz
                    EW[i,j] += Π[j,jp] * (1-job_dest(z_grid[jp],w_grid[i],pars)) * W_init[i,jp] + Π[j,jp] * job_dest(z_grid[jp],w_grid[i],pars) * U_init[i,jp]
                end
                W_new[i,j] = utility(w_grid[i],pars) + β * EW[i,j]
            end
        end 
        ### Posting decision ### 
        for i in 1:nw
            for j in 1:nz
                max_posting_value = zeros(nw)
                for ip in 1:nw
                    max_posting_value[ip] = P[ip,j] * W_new[ip,j] + (1 - P[ip,j]) * (0.9 * U_new[ip,j] + 0.1 * U_new[1,j])
                end
                posting_decision[i,j], index = findmax(max_posting_value)
                w_hat[i,j] = w_grid[index]
            end
        end
        ### Value of unemployment ###
        for i in 1:nw
            for j in 1:nz
                for s in 1:ns
                    EU[i,j,s] = 0
                    util = utility(b_grid[i],pars) - search_costs(s_grid[s],pars) 
                    λ = job_opp(s_grid[s],pars)
                    for jp in 1:nz
                        EU[i,j,s] += Π[j,jp] * λ * posting_decision[i,j] + Π[j,jp] * (1 - λ) * (0.1 * U_new[1,jp] .+ 0.9 * U_new[i,jp])
                    end
                    final_util[s] = util + β * EU[i,j,s]
                end
                max_value, max_index = findmax(final_util)
                U_new[i,j] = max_value
                search_policy[i,j] = s_grid[max_index]
            end
        end
        ### Check for convergence ###                            
        errorW = maximum(abs.(W_new - W_init))
        errorU = maximum(abs.(U_new - U_init))
        error = max(errorW,errorU)
        if error < toler
            println("--------------------")
            println("Converged in $iter iterations")
            println("Error: $error")
            println("--------------------")
            break
        end
        if iter == maxiter
            println("Maximum number of iterations reached")
        end
        if iter == 1
            println("--------------------")
            println("Iteration: $iter")
            println("Error: $error")
        end
        if iter % 100 == 0
            println("--------------------")
            println("Iteration: $iter")
            println("Error: $error")
        end
        ### Update values ###
        W_init = copy(W_new)
        U_init = copy(U_new)
        iter += 1
    end

    return W_init,U_init,search_policy, w_hat
end

function sim(pars, search_policy, wage_policy)
    (; Np, T, Tb) = pars
    sim_z_grid, sim_Π = income_process(pars)
    sim_b_grid = benefits_grid(pars)
    sim_w_grid = wage_grid(pars)
    test_θ = thetas(J_out,pars)
    θ_out = θ_probs(test_θ,pars)
    sim_wages_and_benefits = zeros(Np,T)
    sim_emp_status = zeros(Np,T)
    match_p, δ_p, b_p = random_matrices(pars)
    sim_productivity = productivity_sim(pars)

    for i in 1:Np
        for t in 1:T
            prod = findfirst(sim_z_grid .== sim_productivity[i,t])
            if t == 1
                sim_wages_and_benefits[i,t] = sim_b_grid[1]
                sim_emp_status[i,t] = 0
            else
                if sim_emp_status[i,t-1] == 0
                    benefits = findfirst(sim_b_grid .== sim_wages_and_benefits[i,t-1])
                    match_probability = θ_out[benefits, prod]
                    if match_p[i,t] < match_probability
                        sim_emp_status[i,t] = 1
                        sim_wages_and_benefits[i,t] = wage_policy[benefits, prod]
                    else
                        sim_emp_status[i,t] = 0
                        if b_p[i,t] == 1.0
                            sim_wages_and_benefits[i,t] = sim_b_grid[1]
                        else
                            sim_wages_and_benefits[i,t] = sim_wages_and_benefits[i,t-1]
                        end
                        benefits = findfirst(sim_b_grid .== sim_wages_and_benefits[i,t])
                    end
                else
                    if δ_p[i,t] == 1
                        sim_emp_status[i,t] = 0
                        index = findfirst(sim_w_grid .== sim_wages_and_benefits[i,t-1])
                        sim_wages_and_benefits[i,t] = sim_b_grid[index]
                    else
                        if sim_productivity[i,t] < sim_wages_and_benefits[i,t-1]
                            sim_emp_status[i,t] = 0
                            index = findfirst(sim_w_grid .== sim_wages_and_benefits[i,t-1])
                            sim_wages_and_benefits[i,t] = sim_b_grid[index]
                            benefits = findfirst(sim_b_grid .== sim_wages_and_benefits[i,t])
                        else                               
                            sim_emp_status[i,t] = 1
                            sim_wages_and_benefits[i,t] = sim_wages_and_benefits[i,t-1]
                        end
                    end
                end
            end
        end
    end
    sim_wages_and_benefits = sim_wages_and_benefits[:,Tb+1:end]
    sim_emp_status = sim_emp_status[:,Tb+1:end]
    unemployment_rate = (sum(sim_emp_status .== 0) / (Np * (T - Tb))) * 100
    time_series_unemployment = zeros(Tb)
    for t in 1:Tb
        time_series_unemployment[t] = (sum(sim_emp_status[:,t] .== 0) / Np ) * 100
    end

    return sim_wages_and_benefits, sim_emp_status, unemployment_rate, time_series_unemployment
end

function random_matrices(pars)
    (;Np, T, Tb, δ, b_end) = pars
    dist = Uniform(0,1)
    δ_dist = Binomial(1, δ)
    b_dist = Binomial(1, b_end)
    match_probs = rand(dist, Np, T)
    destruction_probs = rand(δ_dist, Np, T)
    end_benefit_probs = rand(b_dist, Np, T)
    return match_probs, destruction_probs, end_benefit_probs
end

function productivity_sim(pars)
    (;Np, T, Tb) = pars
    z_grid, Π = income_process(pars)
    prod_sim = zeros(Np,T)
    for i in 1:Np
        prod_sim[i,1] = rand(z_grid)
        for t in 2:T
            state_idx = findfirst(isequal(prod_sim[i,t-1]),z_grid)
            prod_sim[i,t] = z_grid[rand(Categorical(Π[state_idx,:]))]
        end
    end
    return prod_sim
end