using Test
using POMDPs
using POMDPGym
using POMDPTools

# Test MDP construction
mdp = GymPOMDP(GymPOMDP(:CartPole).env, pixel_observations=true)
# mdp = GymPOMDP(:CartPole, pixel_observations = true)

@test mdp.pixel_observations == true
@test mdp.γ == 0.99
@test mdp.actions == [1,2]


# Test basic POMDP interface
@test length(rand(initialstate(mdp))) == 4
@test actions(mdp) == [1,2]
@test actionindex(mdp, 1) == 1
@test actionindex(mdp, 2) == 2
@test isterminal(mdp, rand(initialstate(mdp))) == false
@test discount(mdp) == mdp.γ

# Test gen function 
render(mdp.env)


sp, o, r = gen(mdp, mdp.env.state, 1)
close(mdp.env)
@test length(sp) == 4
@test r isa Real
@test size(o) == (400,600,3)

mdp_nopix = GymPOMDP(:CartPole)
sp, o, r = gen(mdp_nopix, convert(Vector{Float32}, mdp_nopix.env.state), 1)
@test all(o .≈ sp)
render(mdp_nopix)
close(mdp_nopix.env)

# Do 1 episode
h = simulate(HistoryRecorder(), mdp_nopix, RandomPolicy(mdp_nopix))
@test isterminal(mdp_nopix, mdp_nopix.env.state)

# Test out prescribed actions
mdp = GymPOMDP(GymPOMDP(:Pendulum).env, pixel_observations=true, actions = [[-1.], [1.]])
@test mdp.actions == [[-1.], [1.]]
# Commented out tests which produce strange segmentation fault errors.
# (Bugs from Python side or handling of GC in Julia-Python interrop?)
# h = simulate(HistoryRecorder(max_steps = 20), mdp, RandomPolicy(mdp))
# close(mdp.env)
# @test !isterminal(mdp, mdp.env.state)

# Test prescribed render function
my_render(sp) = ones(20,20)
mdp = GymPOMDP(GymPOMDP(:Pendulum).env, pixel_observations = true, actions = [[-1.], [1.]], special_render = my_render)
sp, o, r = gen(mdp, mdp.env.state, [1.])
close(mdp.env)
@test size(o) == (20,20)

