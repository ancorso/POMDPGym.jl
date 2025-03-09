using POMDPGym, POMDPs, Test, PyCall
ale_py = pyimport("ale_py")
POMDPGym.pygym.register_envs(ale_py)

mdp = AtariPOMDP(Symbol("ALE/Pong"), version=:v5)
@test length(actions(mdp)) == 6

s = rand(initialstate(mdp))
@test size(s) == (210, 160, 3)

s, o, r = gen(mdp, s, 1)

@test preproc_atari_frame(s) isa Array{Float32} # An earlier version of POMDPGym.jl use `Array{UInt8}`, but now all observations are converted to Float32 arrays

@test size(o) == (80,80,4,1)


@test o[1] isa Float32
@test s[1] isa Float32
render(mdp)

