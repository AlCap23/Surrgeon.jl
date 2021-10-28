## General nonlinear, not solved surrogate
mutable struct Surrogate <: AbstractSurrogate
    # Original Function
    f::Function
    # Incidence
    inc::BitVector
    # Linearities
    linears::BitVector

    # Transformations
    h::Transformation
    op::Function

    # Add tree structure
    parent::AbstractSurrogate
    left::AbstractSurrogate
    right::AbstractSurrogate

    function Surrogate(f::Function, x::AbstractMatrix; kwargs...)
        # Create the incidence of a surrogate
        inc = create_incidence(f, x; kwargs...)
        # Check for linearity
        lins = create_linearity(f, x; kwargs...) .* inc
        # If the surrogate is just linear, return a linear surrogate
        inc == lins && return LinearSurrogate(f, x; kwargs...)
        
        # Preprocess function to be callable with vector / matrix
        f_ = preprocess_function(f)

        return new(
            f_, 
            inc,
            lins, 
            Transformation()
            )
    end
end

function AbstractTrees.children(s::Surrogate)
    if isdefined(s, :left) && !(s===s.left)
        if isdefined(s, :right) !(s===s.right)
            return (s.left, s.right)
        end
        return (s.left,)
    end
    (isdefined(s, :right) && !(s===s.right)) && return (s.right,)
    return ()
end

has_children(s::Surrogate) = (isdefined(s, :left) && !(s === s.left)) || (isdefined(s, :right) && !(s === s.right))
is_solved(s::Surrogate) = has_children(s) ? all(map(is_solved, children(s))) : false 
params(s::Surrogate) = has_children(s) ? map(params, children(s)) : []


function (s::Surrogate)(x, p = params(s))
    if has_children(s) && isdefined(s, :op)
        return reduce(s.op, map(c->c(x, params(c)), children(s)))
    end
    return s.f(x)
end

function leftchild!(s::Surrogate, left::AbstractSurrogate)
    left.parent = s
    s.left = left
    return
end

function rightchild!(s::Surrogate, right::AbstractSurrogate)
    right.parent = s
    s.right = right
    return
end

function set_op!(s::Surrogate, op)
    s.op = op
end

function remove_right!(s::Surrogate)
    setfield!(s, :right, s)
end

function remove_left!(s::Surrogate)
    setfield!(s, :left, s)
end


# Iterations

AbstractTrees.printnode(io::IO, s::Surrogate) = print(io, 
    "Surrogate with $(sum(s.inc)) independent variables. Status : $(is_solved(s))")


## Linear (solved) surrogate
mutable struct LinearSurrogate{T} <: AbstractSurrogate where T <: Number
    # Parameters
    weights::AbstractVector{T}
    bias::T

    # Incidence and linearity
    inc::BitVector
    linears::BitVector
    
    # Coefficient of determination
    determination::T

    # Output transforms
    h::Transformation

    # Parent, if needed
    parent::AbstractSurrogate

    function LinearSurrogate(w::AbstractVector{T}, bias::T, 
        inc::BitVector, linears::BitVector, determination::T = zero(T),
        h::Transformation = Transformation(), parent = nothing
        ) where T <: Number

        obj =  new{eltype(w)}(w, bias, inc, linears, determination, h)

        if !isa(parent, AbstractSurrogate)
            obj.parent = obj
        else
            obj.parent = parent
        end
        return obj

    end

    function LinearSurrogate(f::Function, x::AbstractMatrix{T}; kwargs...) where T <: Number
        inc = create_incidence(f, x)
        lins = Surrgeon.create_linearity(f, x) .* inc
    
        inc != lins  && return Surrogate(f,x; kwargs...)
    
        jac = Surrgeon._gradient(f)(mean(x, dims =2)[:,1]) .* lins .* inc
        y = reduce(hcat, map(f, eachcol(x)))
        bias = mean(y-jac'*x)
        rsq = one(T) .- sum(abs2, y .- bias - jac'x) ./ cov(y, dims = 2)
    
        obj =  new{eltype(x)}(
            jac[:,1], bias, inc, lins, first(rsq), Transformation()
        )

        obj.parent = obj
        
        return obj
    end

    
end



(s::LinearSurrogate)(x::AbstractVector, p = params(s)) = [dot(p[1:end-1],x) + p[end]]
(s::LinearSurrogate)(x::AbstractMatrix, p = params(s)) = reduce(hcat, map(xi->s(xi, p), eachcol(x)))

is_solved(::LinearSurrogate) = true
AbstractTrees.children(::LinearSurrogate) = ()
has_children(::LinearSurrogate) = false
params(x::LinearSurrogate) = vcat(getfield(x, :weights), getfield(x, :bias))

AbstractTrees.printnode(io::IO, s::LinearSurrogate) = print(io, 
    "Linear surrogate with $(sum(s.inc)) independent variables. Status : $(is_solved(s))")

