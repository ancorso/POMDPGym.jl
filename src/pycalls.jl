# Taken from: https://github.com/JuliaML/OpenAIGym.jl/blob/master/src/OpenAIGym.jl

# --------------------------------------------------------------

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
mutable struct GymEnv{T}
    name::Symbol
    ver::Symbol
    pyenv::PyObject   # the python "env" object
    pystep::PyObject  # the python env.step function
    pyreset::PyObject # the python env.reset function
    pystate::PyObject # the state array object referenced by the PyArray state.o
    pystepres::PyObject # used to make stepping the env slightly more efficient
    info::PyObject    # store it as a PyObject for speed, since often unused
    state::T
    reward::Float64
    total_reward::Float64
    actions
    done::Bool
    function GymEnv{T}(name, ver, pyenv, pystate, state) where T
        env = new{T}(name, ver, pyenv, pyenv."step", pyenv."reset",
                                 pystate, PyNULL(), PyNULL(), state)
        env
    end
end

use_pyarray_state(envname::Symbol) = !(envname ∈ (:Blackjack,))

function GymEnv(name::Symbol, ver::Symbol = :v0, render_mode = "rgb_array";
                stateT = ifelse(use_pyarray_state(name), Array{Float32}, Any), kwargs...) # Current version of PyCall seems to do some automatic conversion, so no need to use `PyArray` instead of Julia `Array`
    # PySoccer-related code has been deleted
    GymEnv(name, ver, pygym.make("$name-$ver", render_mode = render_mode; kwargs...), stateT)
end

GymEnv(name::AbstractString; kwargs...) =
    GymEnv(Symbol.(split(name, '-', limit = 2))...; kwargs...)

function GymEnv(name::Symbol, ver::Symbol, pyenv, stateT)
    pystate = pycall(pyenv."reset", PyObject)[1] # Change from earlier version of Gym: a tuple with additional
    # information is returned, so we take `[1]` to get the actual state
    state = convert(stateT, pystate)
    T = typeof(state)
    GymEnv{T}(name, ver, pyenv, pystate, state)
end

function Base.show(io::IO, env::GymEnv)
  println(io, "GymEnv $(env.name)-$(env.ver)")
  if hasproperty(env.pyenv, :class_name)
    println(io, "  $(env.pyenv.class_name())")
  end
  println(io, "  r  = $(env.reward)")
  print(  io, "  ∑r = $(env.total_reward)")
end

# --------------------------------------------------------------

"""
    close(env::GymEnv)
"""
Base.close(env::GymEnv) =
	!ispynull(env.pyenv) && env.pyenv.close()

# --------------------------------------------------------------

"""
    render(env::GymEnv; mode = :human)
# Arguments
- `mode`: `:human`, `:rgb_array`, `:ansi`
"""
render(env::GymEnv, args...; kwargs...) =
    pycall(env.pyenv.render, PyAny; kwargs...)

# --------------------------------------------------------------


function actionset(A::PyObject)
    if hasproperty(A, :n)
        # choose from n actions
		collect(1:convert(Int, A.n))
        # DiscreteSet(0:A.n-1)
    elseif hasproperty(A, :spaces)
        # a tuple of action sets
        sets = [actionset(a) for a in A.spaces]
        TupleSet(sets...)
    elseif hasproperty(A, :high)
        # continuous interval
        [A.low, A.high]
        # if A[:shape] == (1,)  # for now we only support 1-length vectors
        #     IntervalSet{Float64}(A[:low][1], A[:high][1])
        # else
        #     # @show A[:shape]
        #     lo,hi = A[:low], A[:high]
        #     # error("Unsupported shape for IntervalSet: $(A[:shape])")
        #     [IntervalSet{Float64}(lo[i], hi[i]) for i=1:length(lo)]
        # end
    elseif hasproperty(A, :actions)
        # Hardcoded
        collect(1:length(A))
    else
        @show A
        @show keys(A)
        error("Unknown actionset type: $A")
    end
end

pyaction(a::Vector) = Any[pyaction(ai) for ai=a]
pyaction(a) = a

"""
`reset!(env::GymEnv)` reset the environment
"""
function reset!(env::GymEnv)
    pycall!(env.pystate, env.pyreset, PyObject)
    env.pystate = env.pystate[1] # Change from earlier version of Gym: a tuple with additional
    # information is returned, so we take `[1]` to get the actual state
    convert_state!(env)
    env.reward = 0.0
    env.total_reward = 0.0
    env.actions = actionset(env.pyenv.action_space)
    env.done = false
    return env.state
end

function PyObject2Dict(obj)
	ks = collect(obj.keys())
	Dict(k=>obj.get(k) for k in ks)
end
	

"""
    step!(env::GymEnv, a)
take a step in the enviroment
"""
function step!(env::GymEnv, a)
    pyact = pyaction(a)
    pycall!(env.pystepres, env.pystep, PyObject, pyact)

    env.pystate, r, env.done, truncated, env.info = # Newer Gym/Gymnasium added `truncated`
        convert(Tuple{PyObject, Float64, Bool, Bool, PyObject}, env.pystepres)

    convert_state!(env)

    env.total_reward += r
    return (r, env.state, PyObject2Dict(env.info))
end

@inline step!(env::GymEnv, s, a) = step!(env, a)

convert_state!(env::GymEnv{T}) where T =
    env.state = convert(T, env.pystate)

convert_state!(env::GymEnv{<:PyArray}) =
    env.state = PyArray(env.pystate)

finished(env::GymEnv)     = env.done
finished(env::GymEnv, s′) = env.done

# --------------------------------------------------------------

const pygym    = PyNULL()
const pysoccer = PyNULL()

function __init__()
    # the copy! puts the gym module into `pygym`, handling python ref-counting
    copy!(pygym, pyimport("gymnasium"))
end
