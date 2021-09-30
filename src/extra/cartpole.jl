#based off https://github.com/Roboneet/CartPole.jl/blob/master/src/Envs/classic_control/CartPole.jl
@with_kw struct CartPoleMDP <: MDP{Array{Float32}, Array{Float32}}
    gravity::Float32 = 98f-1
    masscart::Float32 = 1f0
    masspole::Float32 = 1f-1
    total_mass::Float32 = masspole + masscart
    length::Float32 = 5f-1 # actually half the pole's length
    polemass_length::Float32 = (masspole * length)
    force_mag::Float32 = 1f1
    τ::Float32 = 2f-2 # seconds between state updates
    maxT::Float32 = τ*100
    kinematics_integrator::AbstractString = "euler"

    # Angle at which to fail the episode
    θ_threshold_radians::Float32 = Float32(12 * 2 * π / 360)
    x_threshold::Float32 = 24f-1
    γ::Float32 = 0.99f0
end

function POMDPs.gen(env::CartPoleMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG)
    x, ẋ, θ, θ̇ , t = s
 
    failed =  !(all(vcat(-env.x_threshold .≤ x .≤ env.x_threshold,
            -env.θ_threshold_radians .≤ θ .≤ env.θ_threshold_radians)))
    timeout = t > env.maxT || t ≈ env.maxT
            
    if failed
        return (sp=fill(-100, size(s)), r=0)
    elseif timeout
        return (sp=fill(100, size(s)), r=0)
    end
    
    
     force = a .* env.force_mag  # action is +1 or -1
     cosθ = cos.(θ)
     sinθ = sin.(θ)
     temp = (force .+ env.polemass_length * θ̇  .^ 2 .* sinθ) / env.total_mass
     θacc = (env.gravity*sinθ .- cosθ.*temp) ./
            (env.length * (4f0/3 .- env.masspole * cosθ .^ 2 / env.total_mass))
     xacc  = temp .- env.polemass_length * θacc .* cosθ / env.total_mass
     if env.kinematics_integrator == "euler"
         x_ = x  .+ env.τ * ẋ
         ẋ_ = ẋ  .+ env.τ * xacc
         θ_ = θ  .+ env.τ * θ̇
         θ̇_ = θ̇  .+ env.τ * θacc
     else # semi-implicit euler
         ẋ_ = ẋ  .+ env.τ * xacc
         x_ = x  .+ env.τ * ẋ_
         θ̇_ = θ̇  .+ env.τ * θacc
         θ_ = θ  .+ env.τ * θ̇_
     end

     sp = vcat(x_, ẋ_, θ_, θ̇_, t+env.τ)
     (sp=sp, r=1f0)
end

function POMDPs.initialstate(mdp::CartPoleMDP)    
    ImplicitDistribution((rng) -> Float32[(rand(Float32, 4) * 1f-1 .- 5f-2)..., 0f0])
end

isfailure(mdp::CartPoleMDP, s) = s[1] == -100
# Terminate if there is a failure or if we exceed the maximum time (in the case that we are tracking such things)
POMDPs.isterminal(mdp::CartPoleMDP, s) = isfailure(mdp, s) || s[1] == 100
POMDPs.discount(mdp::CartPoleMDP) = mdp.γ

