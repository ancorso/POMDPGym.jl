using POMDPGym, POMDPs, Test

mdp = LavaWorldMDP(lava = [GWPos(1,1), GWPos(2,2), GWPos(3,3)], goal=GWPos(4,4))
actions(mdp)
@test length(actions(mdp)) == 4
@test mdp.gridworld.rewards[GWPos(1,1)] == -1
@test mdp.gridworld.rewards[GWPos(2,2)] == -1
@test mdp.gridworld.rewards[GWPos(3,3)] == -1
@test mdp.gridworld.rewards[GWPos(4,4)] == 1
@test GWPos(4,4) in mdp.gridworld.terminate_from
@test GWPos(1,1) in mdp.gridworld.terminate_from
@test GWPos(2,2) in mdp.gridworld.terminate_from
@test GWPos(3,3) in mdp.gridworld.terminate_from

s = rand(initialstate(mdp))
svec = convert_s(AbstractArray, s, mdp) 
@test size(svec) == (7,5,3,1)

s, r = gen(mdp, s, :up)
c = render(mdp, s)

