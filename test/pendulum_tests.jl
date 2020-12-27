using POMDPGym, POMDPs, Test
mdp = PendulumMDP()
actions(mdp)
@test length(actions(mdp)) == 2

s = rand(initialstate(mdp))
@test size(s) == (2,)

c = render(mdp, s, -2)

