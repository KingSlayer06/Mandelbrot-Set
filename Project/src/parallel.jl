# parallel.jl

using Distributed
@everywhere include("sequential.jl") # Including sequential everywhere as all workers use the run_seq() function

#=============================================================================
                            YOUR IMPLEMENTATION HERE
=============================================================================#

# Master 
function generate_fractal_par!(FP::FractalParams, data::Array{Float64, 3})

    N, F, M, center, alpha, delta = extract_params(FP)

    P = nworkers() # P Workers

    # t will be the total time spent in this code
    t = @elapsed begin  
        if P == 0 
            return generate_fractal_seq!(FP, data)
        end

        # Create job and result channels
        job_channel = RemoteChannel(() -> Channel{Int}(P))
        result_channel = RemoteChannel(() -> Channel{Tuple{Int, Matrix{Float64}}}(F))
        
        # Put all jobs in the channel
        for f in 1:F
            frame_data = data[:, :, f]
            put!(result_channel, (f, frame_data))
        end

        for iw in 1:P 
            put!(job_channel, iw)  
        end

        @sync for _ in 1:P
            iw = take!(job_channel) 
            ftr = @spawnat iw work(FP, job_channel, result_channel)
        end

        # Collect results
        for _ in 1:F
            frame_idx, frame_data = take!(result_channel)
            data[:, :, frame_idx] = frame_data
        end
    end
    t
end

# Worker
@everywhere function work(FP::FractalParams, job_channel::RemoteChannel, result_channel::RemoteChannel)

    N, F, max_iters, center, alpha, delta = extract_params(FP)

    worker = myid()

    # Try to get a job
    frame_idx, frame_data = take!(result_channel)
    
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
    
    # Send result back
    put!(result_channel, (frame_idx, frame_data))

    # Indicate availability for new job
    put!(job_channel, worker)
end

#=============================================================================
                            YOUR IMPLEMENTATION HERE
=============================================================================#

function run_par(FP::FractalParams)
    data = zeros(Float64, FP.N, FP.N, FP.F)
    t = generate_fractal_par!(FP, data)
    return t, data
end


