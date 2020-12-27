struct GridWorldMDP <: MDP{GWPos, Symbol}
    g::SimpleGridWorld
end

GridWorldMDP(;size::Tuple{Int, Int}           = (10,10),
              rewards::Dict{GWPos, Float64}   = Dict(GWPos(4,3)=>-10.0, GWPos(4,6)=>-5.0, GWPos(9,3)=>10.0, GWPos(8,8)=>3.0),
              terminate_from::Set{GWPos}      = Set(keys(rewards)),
              tprob::Float64                  = 0.7,
              discount::Float64               = 0.95) = GridWorldMDP(SimpleGridWorld(size = size, rewards = rewards, terminate_from = terminate_from, tprob = tprob, discount = discount))

POMDPs.gen(mdp::GridWorldMDP, s, a, rng = Random.GLOBAL_RNG) = (sp = rand(rng, transition(mdp.g, s, a )), r = POMDPs.reward(mdp.g, s, a))

function POMDPs.initialstate(mdp::GridWorldMDP)
    function istate(rng::AbstractRNG)
        while true
            x, y = rand(rng, 1:mdp.g.size[1]), rand(rng, 1:mdp.g.size[2])
            !(GWPos(x,y) in mdp.g.terminate_from) && return GWPos(x,y)
        end
    end
    return ImplicitDistribution(istate)
end 

POMDPs.convert_s(::Type{AbstractArray}, s::GWPos, mdp::GridWorldMDP) = Float32.([s...])

POMDPs.actions(mdp::GridWorldMDP) = POMDPs.actions(mdp.g)

POMDPs.isterminal(mdp::GridWorldMDP, s) = isterminal(mdp.g, s)
POMDPs.discount(mdp::GridWorldMDP) = discount(mdp.g)

function render(mdp::GridWorldMDP, s=GWPos(7,5), a=nothing; color = s->10.0*POMDPs.reward(mdp.g, s), policy= nothing)
    img = POMDPModelTools.render(mdp.g, (s = s,), color = color, policy = isnothing(policy) ? nothing : FunctionPolicy((s) ->  action(policy, convert_s(AbstractArray, s, mdp))))
    tmpfilename = "/tmp/out.png"
    img |> PNG(tmpfilename, 1cm .* mdp.g.size...)
    load(tmpfilename)
end

