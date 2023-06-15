using DelimitedFiles
using Base.Threads
using BenchmarkTools
using Random

const N = 200
const M = 100_000_000
const L = 10.
const dx = L / N
const dt = round(.1 * (dx)^4; sigdigits=6)
const frames = 10_000
const skip = div(M, frames)

print("T = ", M*dt, '\n')
param_names = ["u, -r", "a", "b", "phi"]


@inline ind(i) = mod(i-1, N)+1
@inline ∇²(A, dx, i) =  (A[ind(i+1)] + A[ind(i-1)] - 2*A[i]) / (dx)^2
@inline ∇(A, dx, i) = (A[ind(i+1)] - A[ind(i-1)]) / (2*dx)

function euler!(
    φ::Array{Float64, 2},
    μ::Array{Float64, 2},
    δφ::Array{Float64, 2},
    ξ::Array{Float64, 2},
    param::NTuple{5, Float64}
    )
    u, α, β, bφ, D = param
    @inbounds for i in axes(φ,1)
        @views ruφ² = u * (-1 + (φ[i, 1]^2 + φ[i, 2]^2 ))
        @views μ[i, 1] = ruφ² * φ[i, 1] - ∇²(φ[:, 1], dx, i) + α * φ[i, 2]
        @views μ[i, 2] = ruφ² * φ[i, 2] - ∇²(φ[:, 2], dx, i) - α * φ[i, 1]
    end
    randn!(ξ)
    @inbounds for i in axes(φ,1)
        @views δφ[i, 1] = ( ∇²(μ[:, 1], dx, i) + D*∇(ξ[:, 1], dx, i) ) * dt
        @views δφ[i, 2] = ( ∇²(μ[:, 2], dx, i) + D*∇(ξ[:, 2], dx, i) ) * dt
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

function loop(φ::Array{Float64, 2}, param::NTuple{5, Float64})
    φt = zeros(frames, N, 2)
    μ = zeros(N, 2)
    δφ = zeros(N, 2)
    ξt = zeros(frames-1, N, 2)
    ξ = zeros(N, 2)
    
    φ[:,1] .+= bφ
    φt[1,:,:] .= φ

    for i in axes(φt, 1)[2:end]
        for j in 1:skip
            euler!(φ, μ, δφ, ξ, param)
            φ .+= δφ
        end
        check(φ, i)
        ξt[i-1,:,:] .= ξ
        φt[i,:,:] .= φ
    end
    
    print('\n')
    return φt, ξt
end

function run_euler(param::NTuple{5, Float64})
    x = LinRange(0, L-dx, N)
    φ = .3 * [sin.(2*pi*x/L) cos.(2*pi*x/L)]
    φt, ξt = loop(φ, param)
    write_file(φt, ξt, param)
    return
end

##############
# Utillities #
##############

write_folder = "data/sym/"

function write_file(φt, ξt, param)
    filename = join(
        param_names[i] * '=' * string(param[i]) * "_"
        for i in range(1, length(param_names))
        )[1:end-1]
    filename = filename * "_N=" * string(N)
    filename = filename * "_dt=" * string(round(dt * skip; sigdigits=5))
    writedlm(write_folder*filename*".txt", reshape(φt, (frames, 2*N)))
    writedlm(write_folder*"noise:"*filename*".txt", reshape(ξt, (frames-1, 2*N )))
end


# we choose r = -us
u = 10.
β = 5. * 1e-12
bφ = -.707
# bφ = -.3
α = 4.
D = sqrt(2 * β / (dx * dt))

param = (u, α, β, bφ, D)


rm(write_folder, recursive=true, force=true)
mkdir(write_folder[1:end-1])

@time run_euler(param);

# αs = LinRange(0, 6, 17)
# φs = [-.8, -1/sqrt(2), -.6, -.5]
# αφ = [(α,φ) for α in αs for φ in φs]
# @time @threads for (α, bφ) in αφ
#     param = (u, bφ, α, β)
#     @time run_euler(param)
# end


 