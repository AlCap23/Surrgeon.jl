# Placeholder for AbstractDifferentiation

function _gradient(f)
    x -> ForwardDiff.gradient(f,x)
end

function _hessian(f)
    x -> ForwardDiff.hessian(f,x)
end

## Analysis

"""
$(SIGNATURES)

Creates the incidence matrix by evaluating the gradient of the function 'f' over each data column of
'x' and normalizes its absolute value by using the 'Inf' norm. Returns a 'BitVector' which indicates if the mean of the 
evaluation of the correspoding index is equal or greater than the relative tolerance 'reltol'.
"""
function create_incidence(f::Function, x::AbstractMatrix; abstol = eps(), reltol = eps(), kwargs...)::BitVector
    jac = _gradient(f)
    ∂ = zeros(eltype(x), size(x, 1))

    for xi in eachcol(x)
        ∂ += normalize!(abs.(jac(xi)), Inf)
    end
    ∂ ./= size(x, 2)

    BitVector(∂ .>= reltol)
end

"""
$(SIGNATURES)

Checks a given function 'f' for linearity by evaluating its gradient over each data column of 'x' and computing
the variance along the data set. Returns a 'BitVector' which indicates if the variance is less or equal than the absolute tolerance 'abstol' and hence 
can be assumed constant.
"""
function create_linearity(f::Function, x::AbstractMatrix; abstol = eps(), reltol = eps(), kwargs...)::BitVector
    jac = _gradient(f)
    return var(map(jac, eachcol(x))) .<= abstol
end

