@with_kw mutable struct RMDP{S,A} <: MDP{S,A}
    env
    π
    cost_fn = POMDPs.reward
    added_states = :none # :none, :time, :acc_reward, :time_and_acc_reward
    dt = 0.1
    maxT = 100*dt
    disturbance_type=:arg #:arg if passed as argument, :noise if used as noise on the state, :action_noise for noise on action
    xscale = 1f0
    rshift = 0f0
    rscale = 1f0
    RMDP(env::MDP{S,A}, π; cost_fn=POMDPs.reward, added_states=:none, dt=0.1, maxT=100*dt, disturbance_type=:arg, xscale=1f0, rshift=0f0, rscale=1f0) where {S,A} = new{S,A}(env, π, cost_fn, added_states, dt, maxT, disturbance_type, xscale, rshift, rscale)
    RMDP(env::POMDP{S, A, O}, π; cost_fn=POMDPs.reward, added_states=:none, dt=0.1, maxT=100*dt, disturbance_type=:arg, xscale=1f0, rshift=0f0, rscale=1f0) where {S,A,O} = new{S,A}(env, π, cost_fn, added_states, dt, maxT, disturbance_type, xscale, rshift, rscale)
end

function POMDPs.initialstate(mdp::RMDP)
    s0 = rand(initialstate(mdp.env))
    mdp.env isa POMDP && (s0 = rand(POMDPs.initialobs(mdp.env, s0)))
    if mdp.added_states in [:time, :acc_reward]
        return ImplicitDistribution((rng) -> [0f0, s0...])
    elseif mdp.added_states == :time_and_acc_reward
        return ImplicitDistribution((rng) -> [0f0, 0f0, s0...])
    elseif mdp.added_states == :none
        return ImplicitDistribution((rng)->s0)
    else
        @error "unrecognized added state: $(mdp.added_states)"
    end
end

function get_s(mdp::RMDP, s)
    if mdp.added_states in [:time, :acc_reward]
        return s[2:end]
    elseif mdp.added_states == :time_and_acc_reward
        return s[3:end]
    else
        return s
    end
end
        
POMDPs.actions(mdp::RMDP) = disturbances(mdp.env)
POMDPs.actionindex(mdp::RMDP, a) = disturbanceindex(mdp.env, a)

function POMDPs.isterminal(mdp::RMDP, s)
    isterm = isterminal(mdp.env, get_s(mdp, s))
    if mdp.added_states in [:time, :time_and_acc_reward]
        isterm = isterm || (s[1] > (mdp.maxT + mdp.dt/2))
    end
    isterm
end

function isfailure(mdp::RMDP, s)
    isfailure(mdp.env, get_s(mdp,s))
end
    
POMDPs.discount(mdp::RMDP) = 1f0

render(mdp::RMDP, s, a = nothing; kwargs...) = render(mdp.env, get_s(mdp,s), a; kwargs...)
render(mdp::RMDP; kwargs...) = render(mdp.env; kwargs...)

function POMDPs.gen(mdp::RMDP, s, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    x = x .* mdp.xscale
    if mdp.env isa MDP
        if mdp.disturbance_type == :arg
            sp, r = gen(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s)), x, rng; kwargs...)
        elseif mdp.disturbance_type == :noise
            sp, r = gen(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x), rng; kwargs...)
        elseif mdp.disturbance_type == :action_noise
            sp, r = gen(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s)) .+ x, rng; kwargs...)
        elseif mdp.disturbance_type == :both
            sp, r = gen(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x[2:end]), x[1], rng; kwargs...)
        else
            @error "Unrecognized disturbance type $(mdp.disturbance_type)"
        end
    elseif mdp.env isa POMDP
        if mdp.disturbance_type == :arg
            sp, o, r = gen(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s)), x, rng; kwargs...)
        elseif mdp.disturbance_type == :noise
            sp, o, r = gen(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x), rng; kwargs...)
        elseif mdp.disturbance_type == :action_noise
            sp, o, r = gen(mdp.env, get_s(mdp, s), clamp.(action(mdp.π, get_s(mdp, s)) .+ x, -1f0, 1f0), rng; kwargs...)
        elseif mdp.disturbance_type == :both
            sp, o, r = gen(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x[2:end]), x[1], rng; kwargs...)
        else
            @error "Unrecognized disturbance type $(mdp.disturbance_type)"
        end
        sp = o
    end
        
    r = (r + mdp.rshift)*mdp.rscale
    
    if mdp.added_states == :time
        sp = [s[1]+mdp.dt, sp...]
    elseif mdp.added_states == :acc_reward
        sp = [s[1] + r, sp...]
    elseif mdp.added_states == :time_and_acc_reward
        sp = [s[1]+mdp.dt, s[2] + r, sp...]
    end
    (sp=sp, r=mdp.cost_fn(mdp, s, sp))
end

function POMDPs.transition(mdp::RMDP, s, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    if mdp.disturbance_type == :arg
        t = transition(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s)), x, rng; kwargs...)
    elseif mdp.disturbance_type == :noise
        t = transition(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x), rng; kwargs...)
    elseif mdp.disturbance_type == :both
        t = transition(mdp.env, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x[2:end]), x[1], rng; kwargs...)
    else
        @error "Unrecognized disturbance type $(mdp.disturbance_type)"
    end
    t
end

POMDPs.reward(mdp::RMDP, s, sp) = mdp.cost_fn(mdp.env, s, sp)

