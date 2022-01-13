
function init_continual_world()
	py"""
		from continualworld.envs import get_single_env
		"""
end
	
function get_clworld_task(name)
	pyenv = py"get_single_env"(name)
	pystate = pycall(pyenv."reset", PyObject)
	state = convert(PyArray, pystate)
	T = typeof(state)
    GymPOMDP(GymEnv{T}(Symbol(name), :v2, pyenv, pystate, state))
end

cw10() = ["hammer-v2",
        "push-wall-v2",
        "faucet-close-v2",
        "push-back-v2",
        "stick-pull-v2",
        "handle-press-side-v2",
        "push-v2",
        "shelf-place-v2",
        "window-close-v2",
        "peg-unplug-side-v2"]

