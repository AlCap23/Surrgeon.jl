using Metatheory

## Functions and simplification rules 

neg(x::T) where {T <: Real} = -x
square(x::T) where {T <: Real} = x^2

const TRANSFORM_THEORY = @theory x begin
    # Exp and log
    log(exp(x)) --> x
    exp(log(x)) --> x
    exp(neg(x)) --> inv(exp(x))
    log(inv(x)) --> neg(log(x))
    
    inv(square(x)) --> square(inv(x))
    inv(sqrt(x)) --> sqrt(inv(x))
    inv(inv(x)) --> x
    inv(neg(x)) --> neg(inv(x))
    
    sqrt(square(x)) --> x
    square(neg(x)) --> square(x)
    neg(neg(x)) --> x
    -x --> neg(x)
    x^2 --> square(x)
end

# Returns the inverse of a function
# We assume the specific function set inv, exp, log, sqrt, square, neg right now
f_inv(x) = x
f_inv(::typeof(exp)) = log
f_inv(::typeof(log)) = exp
f_inv(::typeof(sqrt)) = square
f_inv(::typeof(square)) = sqrt

f_inv(::typeof(+)) = -
f_inv(::typeof(-)) = +
f_inv(::typeof(*)) = /
f_inv(::typeof(/)) = *

## Type

mutable struct Transformation <: AbstractTransformation
    transforms::AbstractVector{F} where F <: Function
end

Transformation() = Transformation(Function[])

(t::Transformation)(x::T) where T <: Number = isempty(t.transforms) ? x : ∘(t.transforms...)(x)

Base.push!(t::Transformation, f::Function) = push!(t.transforms, f)
Base.pushfirst!(t::Transformation, f::Function) = pushfirst!(t.transforms, f)
Base.deleteat!(t::Transformation, i) = deleteat!(t.transforms, i)
deletelast!(t::Transformation) = deleteat!(t.transforms, length(t.transforms))

function simplify!(t::Transformation)
    length(t.transforms) <= 1 && return
    ex = ops_to_expr(t.transforms)
    gr = EGraph(ex)
    saturate!(gr, TRANSFORM_THEORY)
    ex =  extract!(gr, astsize)
    op = eval.(expr_to_ops(ex))
    if length(op) <= 1
        t.transforms = Function[]
    else
        t.transforms = Function[i for i in op[1:end-1]]
    end
end

function invert!(t::Transformation)
    for i in eachindex(t.transforms)
        t.transforms[i] = f_inv(t.transforms[i])
    end
    return
end


function ops_to_expr(op::AbstractVector{Function})
    s = Expr(:call, Symbol(last(op)), :(x))
    for o in reverse(op[1:end-1])
        s = Expr(:call, Symbol(o), s)
    end
    s
end

function expr_to_ops(ex::Expr)
    return [first(ex.args);
        reduce(vcat, map(expr_to_ops, ex.args[2:end]))
    ]
end

function expr_to_ops(::Symbol)
    return
end


#function seperate(f::Function, x::AbstractVector, op::Function = -)
#    
#    #_f = deepcopy(f)
#
#    (x) -> begin
#        _x = similar(x)
#        _x .= x
#        _x[i] = 5.0
#        _f(_x)
#    end
#
#    function h(x)
#        op(_f(x), g(x))
#    end
#
#    return g, h
#end
#
#f(x) = exp(sum(x))
#log(f(x)) = sum(x)
#
#f(x) = g(x[1,2]) + h(x[2,3])