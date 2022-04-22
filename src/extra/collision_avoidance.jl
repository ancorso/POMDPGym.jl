@with_kw struct CollisionAvoidanceMDP <: MDP{Array{Float32}, Float32}
    ddh_max::Float64 = 1.0 # vertical acceleration limit [m/s²]
    collision_threshold::Float64 = 50.0 # collision threshold [m]
    reward_collision::Float64 = -100.0 # reward obtained if collision occurs
    reward_change::Float64 = -1 # reward obtained if action changes
    actions::Vector{Float64} = [-5.0, 0.0, 5.0] # Available actions to command for vertical rate [m/s]
    px = DiscreteNonParametric([2.0, 0.0, -2.0], [0.25, 0.5, 0.25])
    
    h0_dist = Distributions.Uniform(-200, 200)
    dh0_dist  = Distributions.Uniform(-10,10)
    a_prev0 = 0.0
    τ0 = 40
end

# function POMDPs.gen(mdp::CollisionAvoidanceMDP, s, a, x=rand(mdp.px), rng::AbstractRNG = Random.GLOBAL_RNG; )
#     t = transition(mdp, s, a)
#     x_index = findfirst(mdp.px.support .== x)
#     (sp=t.vals[x_index], r = reward(mdp, s, a))
# end

function POMDPs.gen(mdp::CollisionAvoidanceMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG)
    t = transition(mdp, s, a)
    (sp = rand(t), r = reward(mdp, s, a))
end

function POMDPs.transition(mdp::CollisionAvoidanceMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG)
    h, dh, a_prev, τ = s

    # Update the dynamics
    h = h + dh
    # if a != 0.0
    if abs(a - dh) < mdp.ddh_max
       dh += a - dh
    else
       dh += sign(a - dh)*mdp.ddh_max
    end
    # end
    a_prev = a
    τ = max(τ - 1.0, -1.0)
    SparseCat([Float32[h, dh+x, a_prev, τ] for x in mdp.px.support], mdp.px.p)
end

function POMDPs.reward(mdp::CollisionAvoidanceMDP, s, a)
    h, dh, a_prev, τ = s
    
    r = 0.0
    if isfailure(mdp, s)
        # We collided
        r += mdp.reward_collision
    end
    if a != a_prev
        # We changed our action
        r += mdp.reward_change
    end
    r
end

POMDPs.convert_s(::Type{Array{Float32}}, v::V where V <: AbstractVector{Float64}, ::CollisionAvoidanceMDP) = Float32.(v)
POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::Array{Float32}, ::CollisionAvoidanceMDP) = Float64.(s)

function POMDPs.initialstate(mdp::CollisionAvoidanceMDP)
    ImplicitDistribution((rng) -> Float32[rand(mdp.h0_dist), rand(mdp.dh0_dist), mdp.a_prev0, mdp.τ0])
end

POMDPs.actions(mdp::CollisionAvoidanceMDP) = mdp.actions
POMDPs.actionindex(mdp::CollisionAvoidanceMDP, a) = findfirst(mdp.actions .== a)

disturbanceindex(mdp::CollisionAvoidanceMDP, x) = findfirst(mdp.mdp.px.support .== x)
disturbances(mdp::CollisionAvoidanceMDP) = mdp.px.support

function isfailure(mdp::CollisionAvoidanceMDP, s)
    h, dh, a_prev, τ = s
    abs(h) < mdp.collision_threshold && abs(τ) < eps()
end

function POMDPs.isterminal(mdp::CollisionAvoidanceMDP, s)
    h, dh, a_prev, τ = s
    τ < 0.0
end

POMDPs.discount(mdp::CollisionAvoidanceMDP) = 0.99


## Here is a solver that gives the optimal policy
struct OptimalCollisionAvoidancePolicy <: Policy
    𝒜
    grid
    Q
end

function OptimalCollisionAvoidancePolicy(mdp::CollisionAvoidanceMDP, hs=range(-200, 200, length=21), dhs = range(-10, 10, length=21), τs = range(0, 40, length=41))
    grid = RectangleGrid(hs, dhs, actions(mdp), τs)

    𝒮 = [[h, dh, a_prev, τ] for h in hs, dh in dhs, a_prev in actions(mdp), τ in τs]

    # State value function
    U = zeros(length(𝒮))

    # State-action value function
    Q = [zeros(length(𝒮)) for a in actions(mdp)]

    # Solve with backwards induction value iteration
    for (si, s) in enumerate(𝒮)
        for (ai, a) in enumerate(actions(mdp))
            Tsa = transition(mdp, s, a)
            Q[ai][si] = reward(mdp, s, a)
            Q[ai][si] += sum(isterminal(mdp, s′) ? 0.0 : Tsa.probs[j]*GridInterpolations.interpolate(grid, U, vec(s′)) for (j, s′) in enumerate(Tsa.vals))
        end
        U[si] = maximum(q[si] for q in Q)    
    end
    return OptimalCollisionAvoidancePolicy(actions(mdp), grid, Q)
end

function POMDPs.action(policy::OptimalCollisionAvoidancePolicy, s)
    a_best = first(policy.𝒜)
    q_best = -Inf
    for (a,q) in zip(policy.𝒜, policy.Q)
        q = GridInterpolations.interpolate(policy.grid, q, s)
        if q > q_best
            a_best, q_best = a, q
        end
    end
    return a_best
end