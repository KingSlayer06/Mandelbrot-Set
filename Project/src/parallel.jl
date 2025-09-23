# parallel.jl

using Distributed
@everywhere include("sequential.jl") # Including sequential everywhere as all workers use the run_seq() function

#=============================================================================
                            YOUR IMPLEMENTATION HERE
=============================================================================#

# Master
function generate_fractal_par!(FP::FractalParams, data::Array{Float64, 3})

    N, F, M, center, alpha, delta = extract_params(FP)

    P = nworkers()

    if P == 0
        return generate_fractal_seq!(FP, data)
    end

    @assert size(data) == (N, N, F)  "Size mismatch: data has size $(size(data)), but expected ($N, $N, $F)"

    t = @elapsed begin
        job_channel = RemoteChannel(()->Channel{Union{Int,Nothing}}(F))
        result_channel = RemoteChannel(()->Channel{Tuple{Int, Array{Float64,2}}}(F))

        # Producer: push all frame indices into the job channel
        @sync begin
            @async begin
                for f in 1:F
                    put!(job_channel, f)
                end

                for _ in 1:P
                    put!(job_channel, nothing)
                end
            end

            # Spawn exactly P tasks on available workers; each worker will loop taking jobs until `nothing`
            for w in workers()
                @spawnat w work(FP, job_channel, result_channel)
            end

            # Collect results as they arrive and place them into `data`
            received = 0
            while received < F
                idx, frame = take!(result_channel)
                data[:, :, idx] = frame
                received += 1
            end
        end
    end

    t
end

@everywhere function work(FP::FractalParams, job_channel::RemoteChannel, result_channel::RemoteChannel)
    N, F, max_iters, center, alpha, delta = extract_params(FP)

    while true
        job = take!(job_channel)
        # `nothing` is the sentinel to stop
        if job === nothing
            break
        end

        frame_idx = job
        frame_data = zeros(Float64, N, N)

        # Calculate frame-specific parameters
        local_delta = delta * alpha^(frame_idx - 1)
        x_min = center[1] - local_delta
        y_min = center[2] - local_delta
        dw = (2 * local_delta) / N

        # Compute frame
        @inbounds for j in 1:N
            y = y_min + (j - 1) * dw
            @inbounds for i in 1:N
                x = x_min + (i - 1) * dw
                c = Complex(x, y)
                frame_data[i, j] = mandelbrot(c, max_iters)
            end
        end

        # Send index + result back
        put!(result_channel, (frame_idx, frame_data))
    end
end

#=============================================================================
                            YOUR IMPLEMENTATION HERE
=============================================================================#

function run_par(FP::FractalParams)
    data = zeros(Float64, FP.N, FP.N, FP.F)
    t = generate_fractal_par!(FP, data)
    return t, data
end


