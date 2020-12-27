using POMDPGym, POMDPs, Test
mdp = GridWorldMDP()
actions(mdp)
@test length(actions(mdp)) == 4

s = rand(initialstate(mdp))
@test size(s) == (2,)

s, r = gen(mdp, s, :up)
c = render(mdp, s)

