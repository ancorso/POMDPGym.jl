@with_kw mutable struct SequenceMDP{S, A} <: MDP{S,A}
    mdps::Array{MDP{S,A}, 1} # Sequence of mdps to play
    Ns::Array{Int} # Number of experience samples for each mdp before switching to the next one
    count::Int = 0 # current count of the gen function
    logging = false
end

function get_index(Ns, count)
    Nsum = cumsum(Ns)
    loc = count % Nsum[end]
    i = findfirst(loc .< Nsum)
end

function curr(mdps::SequenceMDP)
    mdps.mdps[get_index(mdps.Ns, mdps.count)]
end

POMDPs.convert_s(t::Type{AbstractArray}, s::A2, mdp::SequenceMDP) where A2 = convert_s(t, s, curr(mdp))
POMDPs.convert_s(t::Type{AbstractArray}, s::A2, mdp::SequenceMDP) where {A2 <: AbstractArray} = convert_s(t, s, curr(mdp))

POMDPs.initialstate(mdp::SequenceMDP) = initialstate(curr(mdp))
POMDPs.isterminal(mdp::SequenceMDP, s) = isterminal(curr(mdp), s)
POMDPs.discount(mdp::SequenceMDP) = discount(curr(mdp))
function POMDPs.gen(mdp::SequenceMDP, s, a, args...; kwargs...)
    if !mdp.logging
        mdp.count += 1
    end
    if mdp.count % 1001 == 0
        println("====count: ", mdp.count, " mdp: ", get_index(mdp.Ns, mdp.count))
    end
    gen(curr(mdp), s, a, args...; kwargs...)
end
render(mdp::SequenceMDP, args...; kwargs...) = render(curr(mdp), args...; kwargs...)

