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

function GymPOMDP(env::GymEnv; pixel_observations = false, 
                    γ = 0.99, actions = nothing, special_render = nothing, sign_reward = false, frame_stack = 1, frame_stack_dim = 3, kwargs...)
    if isnothing(actions)
        actions = actionset(env.pyenv.action_space)
    end
    GymPOMDP(env, pixel_observations, γ, actions, special_render, sign_reward, frame_stack, frame_stack_dim)
end

function GymPOMDP(environment::Symbol; version::Symbol = :v1, render_mode = "rgb_array", kwargs...)
    env = GymEnv(environment, version, render_mode, kwargs...)
    GymPOMDP(env; kwargs...)
end

function POMDPs.initialstate(mdp::GymPOMDP, rng::AbstractRNG = Random.GLOBAL_RNG)
    ImplicitDistribution((rng) -> reset!(mdp.env))
end
POMDPs.initialobs(mdp::GymPOMDP, s) = Deterministic(stack_obs(mdp, [rand(observation(mdp, s)) for _=1:mdp.frame_stack]))

POMDPs.actions(mdp::GymPOMDP) = mdp.actions
POMDPs.actionindex(mdp::GymPOMDP, a) = a

function POMDPs.isterminal(mdp::GymPOMDP, s)
    return mdp.env.done
end

POMDPs.discount(mdp::GymPOMDP) = mdp.γ

function POMDPs.observation(mdp::GymPOMDP, s)
    if mdp.pixel_observations
        o = isnothing(mdp.special_render) ? permutedims(Float32.(channelview(render(mdp))), [2,3,1]) : mdp.special_render(s)
    else
        o = convert(Array{Float32}, s)
    end
    Deterministic(o)
end

torgb(o) = collect(colorview(RGB, permutedims(reinterpret.(N0f8, o), [3,1,2])))

function init_mujoco_render()
    py"""
    from mujoco_py import GlfwContext
    GlfwContext(offscreen=True) 
    """
end

function render(mdp::GymPOMDP, s, a = nothing; kwargs...)
    @assert Float32.(s) == Float32.(mdp.env.state)
    render(mdp; kwargs...)
end

render(mdp::GymPOMDP; kwargs...) = torgb(render(mdp.env; kwargs...))

stack_obs(mdp, os) = length(os) == 1 ? os[1] : cat(os..., dims = mdp.frame_stack_dim)

function aggregate_info!(infos, info)
    for k in keys(infos[1])
        info[k] =  mean([i[k] for i in filter((x)->haskey(x, k), infos)])
    end
end

function check_exceeds_steps(mdp)
    if hasproperty(mdp.env.pyenv, :_elapsed_steps)
        if mdp.env.pyenv._elapsed_steps > mdp.env.pyenv._max_episode_steps
            @error "Exceeded timesteps $(mdp.env.pyenv._elapsed_steps) / $(mdp.env.pyenv._max_episode_steps)"
        else
            return
        end
    elseif hasproperty(mdp.env.pyenv, :num_steps)
        if mdp.env.pyenv.num_steps >= mdp.env.pyenv.steps
            @error "Exceeded timesteps $(mdp.env.pyenv.num_steps) / $(mdp.env.pyenv.steps)"
        else
            return
        end
    else
        return
    end
end
    
function POMDPs.gen(mdp::GymPOMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
@assert Float32.(s) == convert(Array{Float32}, mdp.env.state)
    a_py = (a isa Int) ? a - 1 : a # Python indexes from 0
    rtot, os, sp = 0, [], nothing
    infos = []
    for i=1:mdp.frame_stack
        r, sp, ret_info = step!(mdp.env, a_py)
        check_exceeds_steps(mdp)
        o = rand(rng, observation(mdp, sp))
        rtot += r
        push!(os, o)
        push!(infos, ret_info)
    end
    aggregate_info!(infos, info)

    
    mdp.sign_reward && (rtot = sign(rtot))
    o = stack_obs(mdp, os)
    return (sp=sp, o=o, r=rtot)
end

