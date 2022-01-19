Vec2 = SVector{2,Float32}
Vec4 = SVector{4,Float32}

struct Circle
    center::Vec2
    radius::Float32
end

Base.in(pt::Vec2, c::Circle) = norm(pt .- c.center) <= c.radius

@with_kw struct ContinuumWorldMDP <: MDP{Vec4, Vec2}
    range::Tuple{Vec2, Vec2} = (Vec2(-1.0, 1.0), Vec2(-1.0, 1.0)) # range of the contiuous gridworld
    v0::Tuple = (Distributions.Uniform(-.01f0, .01f0), Distributions.Uniform(-.01f0, .01f0)) # Initial velocity distribution
    rewards::Dict{Circle, Float32} = Dict(Circle(Vec2(-0.2f0,-0.4f0), 0.1)=>-1f0, Circle(Vec2(-0.5f0,0.5f0), 0.2f0)=>1f0)
    costs::Dict{Circle, Float32} = Dict()
    disturbance = (Normal(0f0,0.02f0), Normal(0f0,0.02f0))
    discount = 0.99
    Δt = 0.1f0
    vel_thresh = 0.05f0 # The fastest that the agent can be moving in the goal location ofr it to count
    vmax = 1f0 # The fasted the agent can move in a given dimension
    R = I # rotation matrix
end

position(s) = Vec2(s[1:2]...)
velocity(s) = Vec2(s[3:4]...)

isfailure(mdp::ContinuumWorldMDP, s) = !isterminal(mdp, s) && (terminal_reward(mdp, s) < 0 || out_of_bounds(mdp, s))

function snap_to_boundary(mdp, s)
    s = max.(Vec4([mdp.range[1][1], mdp.range[2][1], -mdp.vmax, -mdp.vmax]), s)
    s = min.(Vec4([mdp.range[1][2], mdp.range[2][2], mdp.vmax, mdp.vmax]), s)
    s
end

function out_of_bounds(mdp, s)
    pos = position(s)
    pos[1] <= mdp.range[1][1] || pos[1] >= mdp.range[1][2] ||  pos[2] <= mdp.range[2][1] || pos[2] >= mdp.range[2][2]
end

function terminal_reward(mdp, s)
    pos = position(s)
    vel = velocity(s)
    for (k,v) in mdp.rewards
        @assert v!=0f0 # Don't allow zero rewards because we use reward as a stopping condition
        if pos in k
            v < 0 && return v
            v > 0 && norm(vel) < mdp.vel_thresh && return v
        end
    end
    return 0f0
end

move_to_terminal(mdp, s) = out_of_bounds(mdp, s) || terminal_reward(mdp, s) != 0
    
    
function POMDPs.gen(mdp::ContinuumWorldMDP, s, a, rng = Random.GLOBAL_RNG; info=Dict())
    s = Vec4(s[1:4])
    r = POMDPs.reward(mdp, s)
    info["cost"] = cost(mdp, s)
    a = mdp.R * a
    # x = Vec2(rand(mdp.disturbance[1]), rand(mdp.disturbance[2]))
    if move_to_terminal(mdp, s)
        sp = Vec4([-10.,-10.,-10.,-10.]) # terminal
    else
        x, v = position(s), velocity(s)
        
        v = v .+ clamp.(a .+ x, -1, 1).*mdp.Δt
        v = clamp.(v, -mdp.vmax, mdp.vmax)
        
        x = x .+ v.*mdp.Δt
        sp = snap_to_boundary(mdp, Vec4(x..., v...))
    end
    (sp=sp, r=r)
end

function POMDPs.reward(mdp::ContinuumWorldMDP, s)
    pos = position(s)
    out_of_bounds(mdp, pos) && return -1f0
    r = terminal_reward(mdp, s)
    r != 0 && return r
    
    pos = position(s)
    vel = velocity(s)
    for (k,v) in mdp.rewards
        if v>0
            r += -0.01*v*norm(pos .- k.center)
        end
    end
    return r
end

function cost(mdp::ContinuumWorldMDP, s)
    pos = position(s)
    for (k,v) in mdp.costs
        pos in k && return v
    end
    return 0f0
end

function POMDPs.initialstate(mdp::ContinuumWorldMDP)
    function istate(rng::AbstractRNG)
        while true
            x = Vec2(Float32(rand(rng, Distributions.Uniform(mdp.range[1]...))), Float32(rand(rng, Distributions.Uniform(mdp.range[2]...))))
            v = Vec2(Float32(rand(rng, mdp.v0[1])), Float32(rand(rng, mdp.v0[2])))
            s = Vec4(x..., v...)
            !move_to_terminal(mdp, s) && cost(mdp, s) == 0 && return s
        end
    end
    return ImplicitDistribution(istate)
end 

POMDPs.convert_s(::Type{AbstractArray}, s::Vec4, mdp::ContinuumWorldMDP) = Float32.([s...])
POMDPs.isterminal(mdp::ContinuumWorldMDP, s) = all(s[1:4] .== Vec4([-10., -10., -10., -10.]))
POMDPs.discount(mdp::ContinuumWorldMDP) = mdp.discount

function pos2canvas(p)
    p = Vec2(p[1], -p[2])
    (p .+ 1) ./ 2
end

function vel2canvas(p, scale::Float32 =1f0)
    p = Vec2(p[1], -p[2])
    p .* scale ./ 2
end

scale2canvas(v) = v ./ 2
function render(mdp::ContinuumWorldMDP, s=Vec4(0f0, 0f0, 0f0, 0f0), a=nothing; show_rotation=false)
    s = Vec4(s[1:4])
    rewards = []
    for (k,v) in mdp.rewards
        push!(rewards, (context(), circle(pos2canvas(k.center)..., scale2canvas(k.radius)), v <0 ? fill("red") : fill("green"), stroke("black")))
    end
    
    costs = []
    for (k,v) in mdp.costs
        push!(costs, (context(), circle(pos2canvas(k.center)..., scale2canvas(k.radius)), fill("blue"), stroke("black")))
    end

    pos = pos2canvas(position(s))
    
    l1 = pos2canvas([0,1])
    center = pos2canvas([0,0])
    
    curr_cost = cost(mdp, s)
    extra = []
    if curr_cost > 0
        push!(extra, (context(), circle(pos..., 0.02), stroke("red")))
    end
    
    vel = vel2canvas(velocity(s), mdp.Δt*3)
    acc = isnothing(a) ? nothing : vel2canvas(a, mdp.Δt*3)
    img = compose(context(), 
        (context(), circle(pos..., 0.01), fill("blue"), stroke("black")),
        extra...,
        (context(), line([[pos, pos .+ vel]]), stroke("grey")), isnothing(a) ? nothing :
        (context(), line([[pos .+ vel, pos .+ vel .+ acc]]), stroke("purple")),
        show_rotation ? (context(), line([[center, center .+ mdp.R * l1]]), stroke("orange")) : nothing,
        rewards...,
        costs...,
        (context(), rectangle(), fill("white"))
    )
    tmpfilename = tempname()
    img |> PNG(tmpfilename, 10cm, 10cm)
    load(tmpfilename)
end

