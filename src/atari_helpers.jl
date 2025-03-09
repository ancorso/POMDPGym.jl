function preproc_atari_frame(s)
    r1 = 36:2:194
    r2 = 1:2:160
    v = (s[r1, r2, 1] .+  s[r1, r2, 2] .+ s[r1, r2, 3]) .÷ UInt8(3)
    reshape(v, size(v)..., 1, 1)
end

AtariPOMDP(environment; version = :v0, frame_stack = 4) = GymPOMDP(GymPOMDP(environment, version = version).env, frame_stack = frame_stack, pixel_observations = true, special_render = preproc_atari_frame, sign_reward = true)

