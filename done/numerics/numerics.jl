using Random
using DelimitedFiles
using Base.Threads
using BenchmarkTools
using Tullio
using StaticArrays

using TimerOutputs

tmr = TimerOutput();

const N = 50
const M = 1_000_000
const L = 10
const dx = L / N
const dt = .1 * (dx)^4
const frames = 100
const skip = div(M, frames)

print("T = ", M*dt, '\n')
param_names = ["u, -r", "phibar", "a", "b"]



# Initial conditions
A = 0.1
k = 1


i = range(1, N)
DD = zeros((N, N))
for i in range(1, N)
    DD[i, i] = - 2 / dx^2
    DD[i, i%N + 1] = 1 / dx^2
    DD[i%N + 1, i] = 1 / dx^2
end

D2 = DD
D2dt = DD*dt
eps = [0 .1; -.1 0]


function euler1!(dphi, phi, temp, param)
    u, phibar, a, b = param
    @. dphi = - u * phi
    @. @views dphi += u * (phi[:,1]^2 + phi[:,2]^2 ) * phi
    dphi .-= D2 * phi
    @. @views dphi[:, 1] += a * phi[:, 2]
    @. @views dphi[:, 2] -= a * phi[:, 1]
    randn!(temp)
    @. dphi += temp * b
    dphi .= D2dt * dphi
end

function euler2!(dphi, phi, temp, param)
    u, phibar, a, b = param
    @tullio dphi[x, i] += u*(-1 + ( phi[x, j] * phi[x, j] ) )* phi[x, i]
    @tullio dphi[x, i] += -D2[x, y] * phi[y, i]
    @tullio dphi[x, i] += a*eps[i, j] * phi[x, j]
    randn!(temp)
    @tullio dphi[x,i] += b * temp[x, i]
    @tullio dphi[x, i] = D2[x, y] * dphi[y, i]  * dt
end


function euler3!(dphi, phi, temp, param)
    u, phibar, a, b = param
    @tullio dphi[x, i] += u * (-1 + ( phi[x, j] * phi[x, j] ) ) * phi[x, i]
    dphi .-=  D2 * phi
    @tullio dphi[x, i] += a * eps[i, j] * phi[x, j]
    randn!(temp)
    @. dphi += temp * b
    dphi .= D2 * dphi * dt
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



function loop(param)
    n1 = M//skip
    n2 = n1//10
    
    x = LinRange(0, L - dx, N)
    phi = [(A .* sin.(2*pi*x/L*k) .+ phibar) A .* cos.(2*pi*x/L*k)]
    # phi = zeros(N, 2)
    phit = zeros(div(M, skip), N, 2)
    phit[1,:,:] .= phi
    dphi = zeros(N, 2)
    temp = zeros(N, 2)
    
    for i in axes(phit, 1)[2:end]

        for j in 1:skip
            @timeit tmr "eul1" euler1!(dphi, phi, temp, param)
            # @timeit tmr "eul2" euler2!(dphi, phi, temp, param)
            # @timeit tmr "eul3" euler3!(dphi, phi, temp, param)
            phi .+= dphi
        end
        check(phi, i)
        phit[i,:,:] .= phi
    end
    print('\n')
    return phit
end


function write_file(phit, param)
    filename = join(
        param_names[i] * '=' * string(param[i]) * '_' 
        for i in range(1, length(param_names))
        )[1:end-1]
    writedlm("data/"*filename*".txt", reshape(phit, (frames, 2*N)))
end


function run_euler(param)
    phit = loop(param)
    write_file(phit, param)
end

a = 0.
phibar = -.8
u = 10.
b = 0.1

param = u, phibar, a, b 
@time run_euler(param)
# pprof()

# a = LinRange(4, 6, 12)
# phibar = .5

# function run(a)
#     param = N, M, L, dt, b, r, phibar, a
#     run_euler(param)
# end

# @time @threads for b in a
#     run(b)
# end
 
