# Placeholder for AbstractDifferentiation

function _gradient(f)
    x -> ForwardDiff.gradient(f,x)
end

function _hessian(f)
    x -> ForwardDiff.hessian(f,x)
end