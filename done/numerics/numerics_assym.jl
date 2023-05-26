using DelimitedFiles
using Base.Threads
using BenchmarkTools

const N = 200
const M = 4_000_000
const L = 10.
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
    @inbounds for i in axes(φ,1)
        @views μ[i, 1] = u * (-1 + 2*φ[i, 1]^2) * φ[i, 1] + a * φ[i, 2] - ∇²(φ[:, 1], dx, i) + β*randn(Float64)
        @views μ[i, 2] = u * (-1 + 2*φ[i, 2]^2) * φ[i, 2] - a * φ[i, 1] - ∇²(φ[:, 2], dx, i) + β*randn(Float64) 
    end
    @inbounds for i in axes(φ,1)
        @views δφ[i, 1] = ∇²(μ[:, 1], dx, i) * dt
        @views δφ[i, 2] = ∇²(μ[:, 2], dx, i) * dt
    end
end


function check(φ, i)
    n = frames//10
    if any(isnan, φ)
        throw(ErrorException("Error, NaN detected" ))
    end

    if (div(i,n)) - div(i-1,n) == 1
        print("\r"*"|"^div(i,n))
    end
end

function loop(param::NTuple{4, Float64})
    u, bφ, a, β = param
    nn = 1
    x = collect(LinRange(0, L-dx, N))
    φ = .1 * [sin.(nn*2*pi*x/L) cos.(nn*2*pi*x/L)]
    # φ = zeros(N, 2)
    φt = zeros(frames, N, 2)
    μ = zeros(N, 2)
    δφ = zeros(N, 2)
    
    φ[:,1] .+= bφ
    φt[1,:,:] .= φ

    
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
    φt = loop(param)
    write_file(φt, param)
    return
end

##############
# Utillities #
##############

function write_file(φt, param)
    filename = join(
        param_names[i] * '=' * string(param[i]) * '_' 
        for i in range(1, length(param_names))
        )[1:end-1]
    writedlm("data/assym/"*filename*".txt", reshape(φt, (frames, 2*N)))
end


# we choose r = -us
u = 10.
β = .5
bφ = -0.
α = 1.


# Sol 1
# bφ = 0.
# β = 1.
# u = 2.
# α = 1.
# u = 1.
# α = .5


# Sol 2
# u = 50.
# β = 1.
# bφ = -0.9
# α = 0.

param = (u, bφ, α, β)

# @time run_euler(param);

αs = [0, 2.5, 5., 7.5]
φs = [0, -.2, -.4, -.6, -.8]
αφ = [(α,φ) for α in αs for φ in φs]
@time @threads for (α, bφ) in αφ
    param = (u, bφ, α, β)
    @time run_euler(param)
end


 