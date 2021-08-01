using POMDPGym, POMDPs, Test
pomdp = PendulumPOMDP()
actions(pomdp)
@test length(actions(pomdp)) == 2

s = rand(initialstate(pomdp))
@test size(s) == (2,)

c = render(pomdp, s, -2)


mdp = InvertedPendulumMDP()

