using POMDPGym, POMDPs, Test
mdp = AtariPOMDP(:Pong, version = :v0)
@test length(actions(mdp)) == 6

s = rand(initialstate(mdp))
@test size(s) == (210, 160, 3)

s, o, r = gen(mdp, s, 1)

@test preproc_atari_frame(s) isa Array{UInt8}

@test size(o) == (80,80,4,1)


@test o[1] isa UInt8
@test s[1] isa UInt8
render(mdp)

