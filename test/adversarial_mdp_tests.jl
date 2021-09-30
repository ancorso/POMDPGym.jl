using POMDPGym, POMDPs, Test, Distributions, Random

mdp = ContinuumWorldMDP()
x_dist = MvNormal([0,0], [1,1])

rand(x_dist)
adv_mdp = AdditiveAdversarialMDP(mdp, x_dist)

@test adv_mdp.mdp == mdp
@test adv_mdp.x_distribution isa MvNormal

s = rand(initialstate(adv_mdp))
x = rand(adv_mdp.x_distribution)
a = -x

sp,r = gen(adv_mdp, s, a, x)
sp_2,r_2 = gen(adv_mdp.mdp, s, [0,0])

@test all(sp .≈ sp_2)
@test r ≈ r_2

gen(adv_mdp, s, [0.,0.])

