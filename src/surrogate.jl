function create_incidence(f::Function, x::AbstractMatrix; abstol = eps(), reltol = eps(), kwargs...)
    jac = _gradient(f)
    ∂ = zeros(eltype(x), size(x, 1))

    for xi in eachcol(x)
        ∂ += normalize!(abs.(jac(xi)), Inf)
    end
    ∂ ./= size(x, 2)

    BitVector(∂ .>= reltol)
end

function create_linearity(f::Function, x::AbstractMatrix; abstol = eps(), reltol = eps(), kwargs...)
    jac = _gradient(f)
    return var(map(jac, eachcol(x))) .<= abstol
end

function create_bilinearity(f::Function, x::AbstractMatrix; abstol = eps(), reltol = eps(), kwargs...)
    hes = _hessian(f)
    return hes    
end

struct EmptySurrogate <: AbstractSurrogate end;

is_solved(x::EmptySurrogate) = true

mutable struct Surrogate{F} <: AbstractSurrogate
    # Original Function
    f::F
    # Incidence
    inc::BitVector
    # Linearities
    linears::BitVector

    # Transformations
    #h::AbstractVector{Function}

    # Add children
    op::Function
    left::AbstractSurrogate
    right::AbstractSurrogate

    function Surrogate(f::Function, x::AbstractMatrix; kwargs...)
        inc = create_incidence(f, x; kwargs...)
        lins = create_linearity(f, x; kwargs...) .* inc

        
        inc == lins && return LinearSurrogate(f, x; kwargs...)
                
        function f_(x::AbstractVector, args...; kwargs...)
            f(x)
        end

        function f_(x::AbstractMatrix, args...; kwargs...)
            reduce(hcat, map(f, eachcol(x)))
        end

        return new{typeof(f_)}(f_, 
            inc,
            lins, 
            +, EmptySurrogate(), EmptySurrogate()
            )
    end
end

children(s::Surrogate) = (s.left, s.right)
has_children(s::Surrogate) = any(.! map(x->isa(x, EmptySurrogate), children(s)))
is_solved(s::Surrogate) = has_children(s) ? all(map(is_solved, children(s))) : false 
params(s::Surrogate) = has_children(s) ? map(params, children(s)) : []


function (s::Surrogate)(x, p = params(s))
    if has_children(s)
        return broadcast(s.op, map(c->c(x, params(c)), children(s)))
    end
    return s.f(x)
end


mutable struct LinearSurrogate{T} <: AbstractSurrogate where T <: Number
    weights::AbstractVector{T}
    bias::T

    inc::BitVector
    linears::BitVector
    
    determination::T

    function LinearSurrogate(f::Function, x::AbstractMatrix{T}; kwargs...) where T <: Number
        inc = create_incidence(f, x)
        lins = Surrgeon.create_linearity(f, x) .* inc

        inc != lins  && return Surrogate(f,x; kwargs...)

        jac = Surrgeon._gradient(f)(mean(x, dims =2)[:,1]) .* lins .* inc
        y = reduce(hcat, map(f, eachcol(x)))
        bias = mean(y-jac'*x)
        rsq = one(T) .- sum(abs2, y .- bias - jac'x) ./ cov(y, dims = 2)

        return new{T}(
            jac[:,1], bias, inc, lins, first(rsq)
        )
    end
end

(s::LinearSurrogate)(x, p = params(s)) = p[1:end-1]'x + p[end]

is_solved(::LinearSurrogate) = true
children(::LinearSurrogate) = ()
has_children(::LinearSurrogate) = false
params(x::LinearSurrogate) = vcat(getfield(x, :weights), getfield(x, :bias))
