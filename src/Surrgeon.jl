module Surrgeon

using LinearAlgebra
using Statistics, StatsBase
using AbstractTrees
import AbstractTrees: children

using ForwardDiff

using DocStringExtensions

## Abstract types and minimum function definition

abstract type AbstractSurrogate end;

is_solved(::AbstractSurrogate) = false
AbstractTrees.children(::AbstractSurrogate) = ()
params(::AbstractSurrogate) = []

Base.eltype(::Type{<:TreeIterator{AbstractSurrogate}}) = AbstractSurrogate
Base.IteratorEltype(::Type{<:TreeIterator{AbstractSurrogate}}) = Base.HasEltype()
    


function preprocess_function(f::Function)
    function f_(x::AbstractVector, args...; kwargs...)
        return f(x, args...; kwargs...)
    end

    function f_(x::AbstractMatrix, args...; kwargs...)
        reduce(hcat, map(xi->f_(xi, args...; kwargs...), eachcol(x)))
    end

    return f_
end


include("./ad_backend.jl")

include("./surrogate.jl")
export Surrogate, LinearSurrogate
export children, leftchild!, rightchild!, set_op!

end
