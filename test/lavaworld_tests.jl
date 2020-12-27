using POMDPGym, POMDPs, Test
mdp = LavaWorldMDP()
actions(mdp)
@test length(actions(mdp)) == 4

s = rand(initialstate(mdp))
svec = convert_s(AbstractArray, s, mdp) 
@test size(svec) == (7,5,3,1)

s, r = gen(mdp, s, :up)
c = render(mdp, s)

