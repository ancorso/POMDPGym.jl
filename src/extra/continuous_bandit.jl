@with_kw struct ContinuousBanditMDP <: MDP{Float32, Float32}
    goal::Float32 = 0
    σ::Float32 = 1
end

POMDPs.reward(mdp::ContinuousBanditMDP, s, a) = exp(-(a[1] - mdp.goal)^2 / mdp.σ) - 0.1*abs(a[1] - mdp.goal)
POMDPs.gen(mdp::ContinuousBanditMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict()) = (sp = a[1], r = reward(mdp, s, a))
POMDPs.initialstate(mdp::ContinuousBanditMDP) = Deterministic(Float32(randn()))
POMDPs.isterminal(mdp::ContinuousBanditMDP, s) = true
isfailure(mdp::ContinuousBanditMDP, s) = s[1] > mdp.goal
POMDPs.discount(mdp::ContinuousBanditMDP) = 0.99
POMDPs.convert_s(::Type{V}, s::Float32, mdp::ContinuousBanditMDP) where {V<:AbstractArray} = [s]

