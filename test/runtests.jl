using Test
using POMDPs
using POMDPGym
using POMDPSimulators, POMDPPolicies

# Test MDP construction
mdp = GymPOMDP(:CartPole, pixel_observations = true)

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
sp, o, r = gen(mdp, mdp.env.state, 1)
close(mdp.env)
@test length(sp) == 4
@test r isa Real
@test size(o) == (400,600,3)

mdp_nopix = GymPOMDP(:CartPole, pixel_observations = false)
sp, o, r = gen(mdp_nopix, mdp_nopix.env.state, 1)
@test o == sp

# Do 1 episode
h = simulate(HistoryRecorder(), mdp_nopix, RandomPolicy(mdp_nopix))
@test isterminal(mdp, mdp.env.state)



