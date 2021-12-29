using POMDPGym, POMDPs, Test

env = ZWrap(GymPOMDP(:CartPole), [1.0, 0.5])


s = rand(initialstate(env))
os = [rand(initialobs(env, s))]
i=1
while !isterminal(env, s) && i<100
	s, o, r = gen(env, s, rand([1,2]))
	push!(os, o)
	i += 1
end

@test all([o[1:2] == env.z for o in os])


env = ZWrap(PendulumMDP(), [1.5, 0.5])

s = rand(initialstate(env))
esses = [s]
i=1
while !isterminal(env, s) && i<100
	s, r = gen(env, s, rand(Float32))
	push!(esses, s)
	i += 1
end

@test all([s[1:2] == env.z for s in esses])

