using DelimitedFiles
using Base.Threads

const N = 200
const M = 100_000_000
const L = 10
const dx = L / N
const dt = .1 * (dx)^4
const frames = 1000
const skip = div(M, frames)

print("T = ", M*dt, '\n')
param_names = ["u, -r", "phibar", "a", "b"]


@inline ind(i) = mod(i-1, N)+1
@inline ∇²(A, dx, i) =  (A[ind(i+1)] + A[ind(i-1)] - 2*A[i]) / dx^2

function euler!(
    φ::Array{Float64, 2}, 
    μ::Array{Float64, 2},
    δφ::Array{Float64, 2}, 
    param::NTuple{4, Float64}
    )
    
    u, bφ, a, β = param
    @inbounds for i in 1:N #axes(φ,1)
        @views ruφ² = u * (-1 + (φ[i, 1]^2 + φ[i, 2]^2 ))
        @views μ[i, 1] = ruφ² * φ[i, 1] + a * φ[i, 2] - ∇²(φ[:, 1], dx, i) + β*randn(Float64)
        @views μ[i, 2] = ruφ² * φ[i, 2] - a * φ[i, 1] - ∇²(φ[:, 2], dx, i) + β*randn(Float64) 
    end
    @inbounds for i in 1:N #axes(φ,1)
        @views δφ[i, 1] = ∇²(μ[:, 1], dx, i) * dt
        @views δφ[i, 2] = ∇²(μ[:, 2], dx, i) * dt
    end
end


function check(φ, i)
    n = frames//10
    if sum(isnan.(φ))!=0
        throw(ErrorException("Error, NaN detected" ))
    end

    if (div(i,n)) - div(i-1,n) == 1
        print("\r"*"|"^div(i,n))
    end
end

function loop!(
    φt::Array{Float64, 3}, 
    φ::Array{Float64, 2}, 
    δφ::Array{Float64, 2}, 
    μ::Array{Float64, 2},
    param::NTuple{4, Float64}
    )
    
    for i in axes(φt, 1)[2:end]
        for j in 1:skip
            euler!(φ, μ, δφ, param)
            φ .+= δφ
        end
        check(φ, i)
        φt[i,:,:] .= φ
    end
    print('\n')
    return φt
end


function run_euler(param::NTuple{4, Float64})
    u, bφ, a, β = param
    φ = zeros(N, 2)
    φ[:, 1] .= bφ
    φt = zeros(frames, N, 2)
    φt[1,:,:] .= φ
    μ = zeros(N, 2)
    δφ = zeros(N, 2)
    
    loop!(φt, φ, μ, δφ, param)
    write_file(φt, param)
end

##############
# Utillities #
##############

function write_file(φt, param)
    filename = join(
        param_names[i] * '=' * string(param[i]) * '_' 
        for i in range(1, length(param_names))
        )[1:end-1]
    writedlm("data/"*filename*".txt", reshape(φt, (frames, 2*N)))
end


# we choose r = -u
u = 10.
β = 0.5

# bφ = -1/sqrt(2)
# α = 6.
# param = (u, bφ, α, β)
# @time run_euler(param)

αs = LinRange(0, 6, 17)
φs = [-.8, -1/sqrt(2), -.6, -.5]
αφ = [(α,φ) for α in αs for φ in φs]
@time @threads for (α, bφ) in αφ
    param = (u, bφ, α, β)
    @time run_euler(param)
end


# TODO: How to write for loops? Should i use axes instead of 1:N