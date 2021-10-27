"""
$(SIGNATURES)

Split the `Surrogate` into a linear and nonlinear part assuming that 

f(x) = g(x) + h(x) = g(x) + sum w_i * x_i

holds. g(x) is strictly nonlinear and its support has no intersection with the linear variables x_i forming the linear part.
"""
function linear_split!(s::Surrogate, x::AbstractMatrix; kwargs...)
    sum(s.linears) < 1 && return
    
    # Build linear model h(x) by setting all nonlinear independent variables to a fixed value

    # Find the pivot point

end

"""
$(SIGNATURES)

Find the optimal data column of 'x' which minimizes the influence of the variable indicated via the 'BitVector' 'w' on the function 'f'.
Can be used for additive or multiplicative separability by the correspoding operator '+' or '*' and similar for '-' and '/'.
"""
function find_pivot(f::Function, x::AbstractMatrix, w::BitVector, ::typeof(+); abstol = eps(), reltol = eps())::Int
    
    fmin(x) = abs(abs((x .* w)'Surrgeon._gradient(f)(x)))
    
    best = Inf

    idx = 1

    for i in 1:size(x,2)
        piv = fmin(x[:,i])
        if piv <= best
            best = piv
            idx = i
        end
    end

    return idx
end


function find_pivot(f::Function, x::AbstractMatrix, w::BitVector, ::typeof(*); abstol = eps(), reltol = eps())::Int
    jac = Surrgeon._gradient(f)
    fmin(x) = abs.((w .- .! w)'jac(x) / f(x))

    best = Inf

    idx = 1

    for i in 1:size(x,2)
        piv = fmin(x[:,i])
        if piv <= best
            @show piv x[w,i]
            best = piv
            idx = i
        end
    end

    return idx
end

find_pivot(f::Function, x::AbstractMatrix, w::BitVector, ::typeof(/); kwargs...)::Int = find_pivot(f, x, w, *; kwargs...)
find_pivot(f::Function, x::AbstractMatrix, w::BitVector, ::typeof(-); kwargs...)::Int = find_pivot(f, x, w, +; kwargs...)






