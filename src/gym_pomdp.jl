const O = AbstractArray

struct GymPOMDP{S, A} <: POMDP{S, A, O}
    env::GymEnv{S} # OpenAIGym environment
    pixel_observations::Bool # Whether or not to use an image as the observation
    γ::Float64 # Discount factor
    actions::Vector{A} # List of actions (Discrete only for now)
    special_render # Function for rendering image (takes in env state), nothing if want OpenAI render function
    sign_reward # Whether or not to normalize reward using sign(r)
    frame_stack # number of steps to take and frames to stack to make up observation
    frame_stack_dim
end

function GymPOMDP(environment; version = :v0, pixel_observations = false, 
                    γ = 0.99, actions = nothing, special_render = nothing, sign_reward = false, frame_stack = 1, frame_stack_dim = 3,  kwargs...)
    env = GymEnv(environment, version, kwargs...)
    s0 = reset!(env)
    if isnothing(actions)
        a = OpenAIGym.actions(env, s0)
        a isa LearnBase.DiscreteSet{UnitRange{Int64}} ? (actions = collect(1:length(a))) : error("GymPOMDP Currently only works with discrete action spaces. Got: $(typeof(a))")
    end
    GymPOMDP(env, pixel_observations, γ, actions, special_render, sign_reward, frame_stack, frame_stack_dim)
end

function POMDPs.initialstate(mdp::GymPOMDP, rng::AbstractRNG = Random.GLOBAL_RNG)
    ImplicitDistribution((rng) -> reset!(mdp.env))
end
POMDPs.initialobs(mdp::GymPOMDP, s) = Deterministic(stack_obs(mdp, [rand(observation(mdp, s)) for _=1:mdp.frame_stack]))

POMDPs.actions(mdp::GymPOMDP) = mdp.actions
POMDPs.actionindex(mdp::GymPOMDP, a) = a

POMDPs.isterminal(mdp::GymPOMDP, s) = mdp.env.done
POMDPs.discount(mdp::GymPOMDP) = mdp.γ

function POMDPs.observation(mdp::GymPOMDP, s)
    if mdp.pixel_observations
        o = isnothing(mdp.special_render) ? Float32.(render(mdp)) : mdp.special_render(s)
    else
        o = Float32.(s)
    end
    Deterministic(o)
end

torgb(o) = collect(colorview(RGB, permutedims(reinterpret.(N0f8, o), [3,1,2])))

function render(mdp::GymPOMDP, s, a = nothing)
    @assert s == mdp.env.state
    render(mdp)
end

render(mdp::GymPOMDP) = torgb(OpenAIGym.render(mdp.env, mode = :rgb_array))

stack_obs(mdp, os) = length(os) == 1 ? os[1] : cat(os..., dims = mdp.frame_stack_dim)
    
function POMDPs.gen(mdp::GymPOMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG)
    @assert s == mdp.env.state
    a_py = (a isa Int) ? a - 1 : [a] # Python indexes from 0
    rtot, os, sp = 0, [], nothing
    for i=1:mdp.frame_stack
        r, sp = step!(mdp.env, a_py)
        o = rand(rng, observation(mdp, sp))
        rtot += r
        push!(os, o)
    end
    mdp.sign_reward && (rtot = sign(rtot))
    o = stack_obs(mdp, os)
    return (sp=sp, o=o, r=rtot)
end

