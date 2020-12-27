using POMDPGym, POMDPs, Test
mdp = AtariPOMDP(:Pong, version = :v0)
@test length(actions(mdp)) == 6

s = rand(initialstate(mdp))
@test size(s) == (210, 160, 3)

s, o, r = gen(mdp, s, 1)
@test size(o) == (80,80,4,1)

render(mdp)

