using Random
# using Plots
using DelimitedFiles
using Base.Threads
using BenchmarkTools

N = 70
M = 1_000_000

L = 10
dx = L / N
dt = .4 * (dx)^4
dt = round(dt, digits=10)
frames = 400
skip = div(M, frames)
r = -10
phibar = -.8

print("T = ", M*dt, '\n')

b = .1
# b = b * (N/100)^2

param_names = ["N", "M", "L", "dt", "b", "r", "phibar", "a"]

# Initial conditions
A = 0.2
k = 1


i = range(1, N)
D2 = zeros((N, N))
for i in range(1, N)
    D2[i, i] = - 1 / dx^2
    D2[i, i%N + 1] = 1 / (2*dx^2)
    D2[i%N + 1, i] = 1 / (2*dx^2)
end

function get_x_phi(param)
    N, M, L, dt, b, r, phibar, a = param
    phi = zeros(2, N)
    x = LinRange(0, L - dx, N)
    phi[1,:] .= A .* sin.(2*pi*x/L*k) .+ phibar
    phi[2,:] .= A .* cos.(2*pi*x/L*k)
    return x, phi
end


function f!(dphi, phi, param)
    N, M, L, dt, b, r, phibar, a = param
    @. dphi = r * (1 - (phi[1]^2 + phi[2]^2)) * phi
    dphi .-= phi * D2
    # @. dphi[2,:] += a*phi[1,:]
    # @. dphi[1,:] -= a*phi[2,:]
    # dphi .+= randn((2, N)) * b
    dphi *= D2
    @. dphi *= dt
end

function loop(phit, param, phi)
    n1 = M//skip
    n2 = n1//10
    dphi = zeros(Float64, 2, N)
    for i in range(2, div(M,skip))
        if (div(i,n2)) - div(i-1,n2) == 1
            print("\r"*"|"^div(i,n2))
        end
        for j in range(0, skip)
            f!(dphi, phi, param)
            @. phi += dphi
        end
        phit[i,:,:] .= phi
    end
    print("")
end


function write_file(phit, param)
    filename = join(
        param_names[i] * '=' * string(param[i]) * '_' 
        for i in range(1, length(param_names))
        )[1:end-1]
    writedlm("data/"*filename*".txt", reshape(phit, (frames, 2*N)))
end


function run_euler(param)
    x, phi = get_x_phi(param)

    phit = zeros(Float64, div(M, skip), 2, N)
    phit[1,:,:] .= phi

    loop(phit, param, phi)

    write_file(phit, param)
end


a = 2.
param = N, M, L, dt, b, r, phibar, a
@time run_euler(param)


a = LinRange(4, 6, 10)

function run(a)
    param = N, M, L, dt, b, r, phibar, a
    run_euler(param)
end

# @time @sync for b in a
#     @spawn run(b)
# end

 