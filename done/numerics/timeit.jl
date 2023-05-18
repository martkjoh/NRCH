using TimerOutputs
const tmr = TimerOutput();
struct A
    x
  end
function test(n,x)
    @timeit tmr "set y" y = Vector{A}(undef,n)
    @timeit tmr "loop" for i in 1:n
        @timeit tmr "assign y" y[i] = A(i*x)
    end
    y
end 