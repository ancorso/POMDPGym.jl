@with_kw struct LavaWorld <: MDP{GWPos, Symbol}
    gridworld::SimpleGridWorld # Underlying grdiworld
    lava::Union{Symbol, Array{GWPos}} = [GWPos(1,1)] # Define the lava or select :random
    num_lava_tiles::Int = 1 # Number of squares of lava to use 
    possible_lava_squares::Union{Symbol, Array{GWPos}} = :all # Set of possible lava squares. :all means all of them can be used
    goal::Union{Symbol, GWPos} = GWPos(7,5) # can be :random
    randomize_lava_on_reset::Bool = false # Whether or not to randomize the lava on each reset
    lava_penalty::Float64 = -1. # The reward for landing in lava
    goal_reward::Float64 = 1. # The reward for landing in the goal state
    observation_type = :all # if :all, we get the full grid over 3 channels
    rng::AbstractRNG = Random.GLOBAL_RNG
    initial_state = :random
end

function LavaWorldMDP(;size = (7, 5), tprob = 1.0, discount = 0.95, rewards = nothing, num_lava_tiles::Int = 1,
                        lava::Union{Symbol, Array{GWPos}} = [GWPos(1,1)],
                        goal::Union{Symbol, GWPos} = GWPos(7,5),
                        randomize_lava_on_reset::Bool = false,
                        possible_lava_squares::Union{Symbol, Array{GWPos}} = :all,
                        lava_penalty::Float64 = -1.,
                        goal_reward::Float64 = 1.,
                        observation_type = :all,
                        initial_state = :random,
                        rng::AbstractRNG = Random.GLOBAL_RNG)
         mdp = LavaWorld(gridworld = SimpleGridWorld(size = size, tprob = tprob, discount = discount, rewards = lava == :random ? Dict() : merge(Dict(p=>lava_penalty for p in lava), Dict(goal=>goal_reward))), 
            num_lava_tiles = num_lava_tiles, 
            possible_lava_squares = possible_lava_squares, 
            goal = goal, 
            randomize_lava_on_reset = randomize_lava_on_reset,
            lava_penalty = lava_penalty, 
            goal_reward = goal_reward, 
            observation_type = observation_type,
            rng = rng,
            initial_state = initial_state)
    lava == :random && randomize_lava!(mdp)
    mdp
end

POMDPs.gen(mdp::LavaWorld, s, a, rng = Random.GLOBAL_RNG) = (sp = rand(rng, transition(mdp.gridworld, s, a )), r = POMDPs.reward(mdp.gridworld, s, a))

function random_lava(mdp::LavaWorld)
    size, rng = mdp.gridworld.size, mdp.rng
    g = mdp.goal == :random ? GWPos(rand(rng, 1:size[1]), rand(rng, 1:size[2])) : mdp.goal
    lava = []
    while length(lava) < mdp.num_lava_tiles
        if mdp.possible_lava_squares == :all
            p = GWPos(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        else
            p = rand(rng, mdp.possible_lava_squares)
        end
        p != g && push!(lava, p)
    end 
    @assert !(g in lava)
    rewards = Dict(p => mdp.lava_penalty for p in lava)
    rewards[g] = mdp.goal_reward 
    rewards
end

function randomize_lava!(mdp::LavaWorld)
    g = mdp.gridworld
    [delete!(g.rewards, k) for k in keys(g.rewards)]
    [delete!(g.terminate_from, k) for k in g.terminate_from]
    rs = random_lava(mdp)
    for k in keys(rs)
        g.rewards[k] = rs[k]
        push!(g.terminate_from, k)
    end
end

function POMDPs.initialstate(mdp::LavaWorld)
    mdp.randomize_lava_on_reset && randomize_lava!(mdp)
    
    # return Deterministic(GWPos(1,5))
    if mdp.initial_state == :random
        function istate(rng::AbstractRNG)
            while true
                x, y = rand(rng, 1:mdp.gridworld.size[1]), rand(rng, 1:mdp.gridworld.size[2])
                !(GWPos(x,y) in mdp.gridworld.terminate_from) && return GWPos(x,y)
            end
        end
        return ImplicitDistribution(istate)
    else
        return Deterministic(mdp.initial_state)
    end
end 

POMDPs.actions(mdp::LavaWorld) = [POMDPs.actions(mdp.gridworld)...]
POMDPs.states(mdp::LavaWorld) = states(mdp.gridworld)
POMDPs.reward(mdp::LavaWorld, s) = POMDPs.reward(mdp.gridworld, s)
POMDPs.isterminal(mdp::LavaWorld, s) = isterminal(mdp.gridworld, s)
POMDPs.discount(mdp::LavaWorld) = discount(mdp.gridworld)
            
function POMDPs.convert_s(::Type{V}, s::GWPos, mdp::LavaWorld) where {V<:AbstractArray}
    if mdp.observation_type == :all
        svec = zeros(Float32, mdp.gridworld.size..., 3, 1)
        !isterminal(mdp, s) && (svec[s[1], s[2], 3] = 1.)
        for p in states(mdp)
            POMDPs.reward(mdp, p) < 0 && (svec[p[1], p[2], 2] = 1.)
            POMDPs.reward(mdp, p) > 0 && (svec[p[1], p[2], 1] = 1.)
        end
        return svec
    else
        return [s...]
    end
end

function POMDPs.convert_s(::Type{GWPos}, v::V, mdp::LavaWorld) where {V<:AbstractArray} 
    if mdp.observation_type == :all
        return GWPos(findfirst(reshape(v, mdp.gridworld.size..., :)[:,:,3] .== 1.0).I)
    else
        return v
    end
end

goal(mdp::LavaWorld, s) = GWPos(findfirst(reshape(s, mdp.gridworld.size..., :)[:,:,1] .== 1.0).I)

function gen_occupancy(buffer, mdp)
    N = length(buffer)
    occupancy = Dict(s => 0. for s in states(mdp))
    cols = fill(:, ndims(buffer[:s])-1)
    for i=1:N
        s = convert_s(GWPos, buffer[:s][cols..., i], mdp)
        occupancy[s] += 1 / N
    end
    occupancy 
end

function render(mdp::LavaWorld, s=GWPos(7,5), a=nothing; color = s->10.0*POMDPs.reward(mdp, s), policy= nothing, return_compose = false)
    img = POMDPModelTools.render(mdp.gridworld, (s = s,), color = color, policy = isnothing(policy) ? nothing : FunctionPolicy((s) ->  action(policy, convert_s(AbstractArray, s, mdp))[1]))
    return_compose && return img
    tmpfilename = "/tmp/out.png"
    img |> PNG(tmpfilename, 1cm .* mdp.gridworld.size...)
    load(tmpfilename)
end

# render_and_save(filename, g::MDP...) = hcat_and_save(filename,  [POMDPModelTools.render(gi) for gi in g]...)
# 
# function hcat_and_save(filename, c::Context...)
#     set_default_graphic_size(35cm,10cm)
#     r = compose(Compose.context(0,0,1cm, 0cm), Compose.rectangle()) # spacer
#     cs = []
#     for ci in c
#         push!(cs, ci)
#         push!(cs, r)
#     end
#     hstack(cs[1:end-1]...) |> PDF(filename)
# end

