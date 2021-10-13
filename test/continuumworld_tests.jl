using POMDPGym, POMDPs, Test
mdp = ContinuumWorldMDP()

render(mdp)

for i=1:1000
    @test abs(reward(mdp, rand(initialstate(mdp)))) < 1
end
s = rand(initialstate(mdp))
svec = convert_s(AbstractArray, s, mdp)
@test all(s.==svec) 
@test size(svec) == (4,)


s, r = gen(mdp, s, (0.01, 0.01))
c = render(mdp, POMDPGym.Vec4(0.9, 0.9, 0.05f0, 0.05f0), POMDPGym.Vec2(0.05, 0.05))


mdp_cost = ContinuumWorldMDP(rewards = Dict(Circle(Vec2(-0.5f0,0.5f0), 0.2f0)=>1f0), costs=Dict(Circle(Vec2(-0.2f0,-0.4f0), 0.1)=>1f0))

@test POMDPGym.cost(mdp_cost, POMDPGym.Vec4(0.9, 0.9, 0.05f0, 0.05f0)) == 0
@test POMDPGym.cost(mdp_cost, POMDPGym.Vec4(-0.2, -0.4, 0.05f0, 0.05f0)) == 1

render(mdp_cost, POMDPGym.Vec4(0.9, 0.9, 0.05f0, 0.05f0))
render(mdp_cost, POMDPGym.Vec4(-0.2, -0.4, 0.05f0, 0.05f0))

