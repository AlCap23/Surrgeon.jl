module Surrgeon

using LinearAlgebra
using Statistics, StatsBase
using ForwardDiff

abstract type AbstractSurrogate end;

include("./ad_backend.jl")

include("./surrogate.jl")
export Surrogate
export create_incidence

end
