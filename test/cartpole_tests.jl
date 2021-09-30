using POMDPGym, POMDPs, Test
mdp = CartPoleMDP()

s = rand(initialstate(mdp))
@test length(s) == 5
@test s[end] == 0f0

sp, r = gen(mdp, s, -1)

