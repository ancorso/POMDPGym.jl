# Common functions for dynamics and rendering of pendulums
function pendulum_dynamics(env, s, a, rng::AbstractRNG = Random.GLOBAL_RNG)
    θ, ω = s[1], s[2]
    dt, g, m, l = env.dt, env.g, env.m, env.l

    a = a[1]
    a = clamp(a, -env.max_torque, env.max_torque)
    costs = angle_normalize(θ)^2 + 0.1 * ω^2 + 0.001 * a^2

    ω = ω + (-3. * g / (2 * l) * sin(θ + π) + 3. * a / (m * l^2)) * dt
    θ = θ + ω * dt
    ω = clamp(ω, -env.max_speed, env.max_speed)

    sp = [θ, ω]
    r = env.Rstep - env.λcost*costs
    return sp, r
end

angle_normalize(x) = mod((x+π), (2*π)) - π

function render_pendulum(env, s, a)
    θ = s[1] + π/2.
    point_array = [(0.5,0.5), (0.5 + 0.3*cos(θ), 0.5 - 0.3*sin(θ))]
    
    a_rad = abs(a)/10.
    if a < 0
        θstart = -3π/4
        θend = -θstart
        θarr = θend
    else
        θstart = π/4
        θend = -θstart
        θarr = θstart
    end 
    
    
    # Draw the arrow 
    endpt = (0.5, 0.5) .+ a_rad.*(cos(θarr), sin(θarr))
    uparr = endpt .+ 0.1*a_rad.*(cos(θarr)-sign(a)*sin(θarr), sign(a)*cos(θarr)+sin(θarr))
    dwnarr = endpt .+ 0.1*a_rad.*(-cos(θarr)-sign(a)sin(θarr), sign(a)*cos(θarr)-sin(θarr))
    arrow_array = [[endpt, uparr], [endpt, dwnarr]]
    
    
    img = compose(context(),
        (context(), line(arrow_array), arc(0.5, 0.5, a_rad, θstart, θend),  linewidth(0.5mm), fillopacity(0.), stroke("red")),
        (context(), circle(0.5, 0.5, 0.01), fill("blue"), stroke("black")),
        (context(), line(point_array), stroke("black"), linewidth(1mm)),
        (context(), rectangle(), fill("white"))
    )
    tmpfilename = tempname()
    img |> PNG(tmpfilename, 10cm, 10cm)
    load(tmpfilename)
end

## POMDP implementation -- Allows for pixel observation
@with_kw struct PendulumPOMDP <: POMDP{Array{Float64}, Float64, Array{Float32}}
    failure_thresh::Union{Nothing, Float64} = nothing # if set, defines the operating range fo the pendulum. Episode terminates if abs(θ) is larger than this. 
    θ0 = Distributions.Uniform(-π, π) # Distribution to sample initial angular position
    ω0 = Distributions.Uniform(-1., 1.) # Distribution to sample initial angular velocity
    Rstep = 0 # Reward earned on each step of the simulation
    λcost = 1 # Coefficient to the traditional OpenAIGym Reward
    max_speed::Float64 = 8.
    max_torque::Float64 = 2.
    dt::Float64 = .05
    g::Float64 = 10.
    m::Float64 = 1.
    l::Float64 = 1.
    γ::Float64 = 0.99
    actions::Vector{Float64} = [-1., 1.]
    pixel_observations::Bool = false
    render_fun::Union{Nothing, Function} = nothing
end

InvertedPendulumPOMDP(failure_thresh = deg2rad(20), 
                    θ0 = Distributions.Uniform(-failure_thresh/2., failure_thresh/2.),
                    ω0 = Distributions.Uniform(-.1, .1),
                    Rstep = 1,
                    λcost = 1;
                    kwargs...) = PendulumPOMDP(failure_thresh = failure_thresh, θ0 = θ0, ω0 = ω0, Rstep = Rstep, λcost = λcost; kwargs...)



function POMDPs.gen(mdp::PendulumPOMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG)
    sp, r = pendulum_dynamics(mdp, s, a, rng)
    (sp = sp, o=rand(rng, observation(mdp, sp)), r = r)
end

function POMDPs.observation(mdp::PendulumPOMDP, s)
    o = mdp.pixel_observations ? mdp.render_fun(s) : [angle_normalize(s[1]), s[2]] #[cos(s[1]), sin(s[1]), s[2]]
    Deterministic(Float32.(o))
end

function POMDPs.initialstate(mdp::PendulumPOMDP)
    ImplicitDistribution((rng) -> Float32.([rand(rng, mdp.θ0), rand(rng, mdp.ω0)]))
end

POMDPs.initialobs(mdp::PendulumPOMDP, s) = observation(mdp, s)

POMDPs.actions(mdp::PendulumPOMDP) = mdp.actions

POMDPs.isterminal(mdp::PendulumPOMDP, s) = !isnothing(mdp.failure_thresh) && abs(s[1]) > mdp.failure_thresh
POMDPs.discount(mdp::PendulumPOMDP) = mdp.γ

render(mdp::PendulumPOMDP, s, a::AbstractArray) = render(mdp, s, a...)
render(mdp::PendulumPOMDP, s, a = 0) = render_pendulum(mdp, s, a)

## MDP formulation
@with_kw struct PendulumMDP <: MDP{Array{Float64}, Array{Float32}}
    failure_thresh::Union{Nothing, Float64} = nothing # if set, defines the operating range fo the pendulum. Episode terminates if abs(θ) is larger than this. 
    θ0 = Distributions.Uniform(-π, π) # Distribution to sample initial angular position
    ω0 = Distributions.Uniform(-1., 1.) # Distribution to sample initial angular velocity
    Rstep = 0 # Reward earned on each step of the simulation
    λcost = 1 # Coefficient to the traditional OpenAIGym Reward
    max_speed::Float64 = 8.
    max_torque::Float64 = 2.
    dt::Float64 = .05
    g::Float64 = 10.
    m::Float64 = 1.
    l::Float64 = 1.
    γ::Float64 = 0.99
    actions::Vector{Float64} = [-1., 1.]
end

InvertedPendulumMDP(failure_thresh = deg2rad(20), 
                    θ0 = Distributions.Uniform(-failure_thresh/2., failure_thresh/2.),
                    ω0 = Distributions.Uniform(-.1, .1),
                    Rstep = 1,
                    λcost = 1;
                    kwargs...) = PendulumMDP(failure_thresh = failure_thresh, θ0 = θ0, ω0 = ω0, Rstep = Rstep, λcost = λcost; kwargs...)

function POMDPs.gen(mdp::PendulumMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG)
    sp, r = pendulum_dynamics(mdp, s, a, rng)
    (sp = sp, r = r)
end

function POMDPs.initialstate(mdp::PendulumMDP)
    ImplicitDistribution((rng) -> Float32.([rand(rng, mdp.θ0), rand(rng, mdp.ω0)]))
end

POMDPs.actions(mdp::PendulumMDP) = mdp.actions

POMDPs.isterminal(mdp::PendulumMDP, s) = !isnothing(mdp.failure_thresh) && abs(s[1]) > mdp.failure_thresh
POMDPs.discount(mdp::PendulumMDP) = mdp.γ

render(mdp::PendulumMDP, s, a::AbstractArray) = render(mdp, s, a...)
render(mdp::PendulumMDP, s, a = 0) = render_pendulum(mdp, s, a)