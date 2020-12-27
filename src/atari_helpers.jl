function preproc_atari_frame(s)
    v = sum(s .÷ UInt8(3) , dims = 3)[36:2:194, 1:2:end, :]
    reshape(v, size(v)..., 1)
end

AtariPOMDP(environment; version = :v0, frame_stack = 4) = GymPOMDP(environment, version = version, frame_stack = frame_stack, pixel_observations = true, special_render = preproc_atari_frame, sign_reward = true)

