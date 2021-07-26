using POMDPGym, POMDPs, Test, PyCall

init_mujoco_render() # Required for visualization
pyimport("safety_gym")
mdp = GymPOMDP(Symbol("Safexp-PointGoal1"), version=:v0)


s = rand(initialstate(mdp))
@test size(s) == (60,)


info = Dict()
s, o, r = gen(mdp, s, 1, info=info)

info

