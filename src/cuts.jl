"""
$(SIGNATURES)

Assert the validity of the model split via the `abstol` and `reltol` parameter.
"""
function assert_split(s::Surrogate, x::AbstractMatrix; abstol = eps(), reltol = eps())
    !has_children(s) && return true
    y = s.f(x) # Call original
    e = norm(y - s(x), 2)
    e <= abstol && (e / norm(y, 2)) <= reltol
end

"""
$(SIGNATURES)

Split the `Surrogate` into a linear and nonlinear part assuming that 

f(x) = g(x) + h(x) = g(x) + sum w_i * x_i

holds. g(x) is strictly nonlinear and its support has no intersection with the linear variables x_i forming the linear part.
"""
function linear_split!(s::Surrogate, x::AbstractMatrix; kwargs...)
    sum(s.linears) < 1 && return
    
    # Build linear model h(x) by setting all nonlinear independent variables to a fixed value

    # Compute the linear coefficients
    jac = _gradient(s.f)
    w = zeros(eltype(x), size(x, 1))
    w[s.linears] = mean(map(xi->jac(xi)[s.linears], eachcol(x)))
    
    linmod = LinearSurrogate(
        w, zero(eltype(x)), 
        s.linears, s.linears, 
        zero(eltype(x)), s.h, s
    )

    f_diff = function (x) let f= s.f, w = w
        f(x) - dot(w, x)
    end
    end

    nonlin = Surrogate(f_diff, x; kwargs...)

    rightchild!(s, linmod)
    leftchild!(s, nonlin)
    set_op!(s, +)
    return
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
            best = piv
            idx = i
        end
    end

    return idx
end

find_pivot(f::Function, x::AbstractMatrix, w::BitVector, ::typeof(/); kwargs...)::Int = find_pivot(f, x, w, *; kwargs...)
find_pivot(f::Function, x::AbstractMatrix, w::BitVector, ::typeof(-); kwargs...)::Int = find_pivot(f, x, w, +; kwargs...)

find_pivot(s::Surrogate, args...; kwargs...) = find_pivot(s.f, args...; kwargs...)

"""
$(SIGNATURES)

Tries to split the surrogate model with the given operator to construct a model which would read

f(x) = op(g(x), h(x))

and stores the surrogates g(x) and h(x) as the left and right child. 

The `+` operator would result in 

f(x) = g(x) + h(x)

And the `/` operator in 

f(x) = g(x) / h(x)

If the absolute tolerance or relative tolerance, specified via the keywords `abstol` and `reltol`, of the resulting model is not sufficient, than the original surrogate will
be returned.
"""
function split_by!(s::Surrogate, x::AbstractMatrix, w::BitVector, op::Function; kwargs...)
    idx = find_pivot(s, x, w, op; kwargs...)
    xval = x[w, idx]

    f_left = function (x) let f = s.f, w = w, piv = xval
        _x = deepcopy(x)
        _x[w] = piv
        f(_x)
    end
    end

    f_right = function (x) let f = s.f, g = f_left, inv_op = f_inv(op)
        return broadcast(inv_op, f(x), g(x))
    end
    end

    left = Surrogate(f_left, x)
    right = Surrogate(f_right, x)
    
    leftchild!(s, left)
    rightchild!(s, right)
    set_op!(s, op)

    if assert_split(s, x; kwargs...)
        return
    else
        remove_left!(s)
        remove_right!(s)
    end
end





