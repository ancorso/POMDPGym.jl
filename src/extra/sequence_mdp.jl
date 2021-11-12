@with_kw mutable struct SequenceMDP{S, A} <: MDP{S,A}
    mdps::Array{MDP{S,A}, 1} # Sequence of mdps to play
    Ns::Array{Int} # Number of experience samples for each mdp before switching to the next one
    count::Int = 0 # current count of the gen function
end

function get_index(Ns, count)
    Nsum = cumsum(Ns)
    loc = count % Nsum[end]
    i = findfirst(loc .< Nsum)
end

function curr(mdps::SequenceMDP)
    mdps.mdps[get_index(mdps.Ns, mdps.count)]
end

POMDPs.initialstate(mdp::SequenceMDP) = initialstate(curr(mdp))
POMDPs.isterminal(mdp::SequenceMDP, s) = isterminal(curr(mdp), s)
POMDPs.discount(mdp::SequenceMDP) = discount(curr(mdp))
function POMDPs.gen(mdp::SequenceMDP, s, a, args...; kwargs...)
    mdp.count += 1
    gen(curr(mdp), s, a, args...; kwargs...)
end
render(mdp::SequenceMDP, args...; kwargs...) = render(curr(mdp), args...; kwargs...)

