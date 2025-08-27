using ArgParse
using BlochSimulators
using CompasToolkit
using ComputationalResources
using ImagePhantoms
using TimerOutputs
using JLD2
using LinearAlgebra
using LinearMaps
using Pkg
using PythonPlot
using Random
using Revise
using StaticArrays
using Statistics
using StructArrays
using TrustRegionReflective

include("DerivativeOperations/DerivativeOperations.jl")
using .DerivativeOperations

include("utils/make_phantom.jl")
include("utils/mask.jl")
include("utils/objective.jl")
include("utils/RelaxationColors.jl")
include("utils/pythonplot.jl")


function parse_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--num-slices", "-n"
            help = "number of slices to process"
            arg_type = Int
        "--start-slice", "-s"
            help = "offset slice from where to start processing"
            arg_type = Int
        "--threshold", "-t"
            help = "threshold used for calculating mask"
            arg_type = Float32
            default = 0.05
        "--steihaug-tolerance"
            help = "tolerance for steihaug in range [0, 1]"
            arg_type = Float64
            default = 0.01
        "--convergence-tolerance"
            help = "tolerance for convergence in range [0, 1]"
            arg_type = Float64
            default = 0.01
        "--input", "-i"
            help = "the input file"
            arg_type = String
            default = "mrstat_3d_decoupled_with_pd2.jld2"
        "--output", "-o"
            help = "the output file"
            arg_type = String
            default = "output.jld2"
        "--gc-debug"
            help = "enable garbage collection logging"
            action = :store_true
        "--upsample"
            help = "upsample the spatial resolution by a given factor"
            arg_type = Float64
            default = 1.0
    end

    return ArgParse.parse_args(s)
end


function main(args)
    if args["gc-debug"]
        GC.enable_logging(true)
    end

# Load JLD2 file with data, sequence, trajectory and coordinates
    file_name = args["input"]
    @load file_name data sequence trajectory coordinates pd

    Nx = size(coordinates, 1)
    Ny = size(coordinates, 2)
    nr_slices = size(coordinates, 4)
    ncoils = 1

    println("parsing ", file_name)
    println("- num. slices: ", nr_slices)
    println("- num. voxels: ", Nx, "x", Ny)
    println("- num. readouts: ", trajectory.nreadouts)
    println("- num. samples per readout: ", trajectory.nsamplesperreadout)
    println("- num. coils: ", ncoils)

# Fix k0
    kˣ = -real.(trajectory.Δk_adc) * trajectory.nsamplesperreadout / 2
    kʸ = imag.(trajectory.k_start_readout)
    trajectory.k_start_readout .= kˣ .+ kʸ .* im

    sliceprofiles = ones(length(sequence.RF_train),1) .|> complex;
    unwrap(::Val{x}) where x = x

# Make coil sensitivities
    coil_sensitivities = ComplexF32.(ones(Nx, Ny, ncoils))

# Compas data structures
    compas_context = CompasToolkit.init_context(0)

    compas_sequence = CompasToolkit.FispSequence(
        sequence.RF_train,
        sliceprofiles,
        sequence.TR,
        sequence.TE,
        unwrap(sequence.max_state),
        sequence.TI;
        undersampling_factor=sequence.py_undersampling_factor,
        repetitions=sequence.repetitions,
    )

    compas_trajectory = CompasToolkit.CartesianTrajectory(
        trajectory.nreadouts,
        trajectory.nsamplesperreadout,
        trajectory.Δt,
        trajectory.k_start_readout,
        trajectory.Δk_adc
    )

# Compute mask
    mask = calculate_ρ_mask(pd[:,:,1,:], args["threshold"])

    fraction = sum(mask) / length(mask) * 100
    println("calculated mask, $fraction% is within mask")

# Add noise?
    # TODO

# Set reconstruction options
    trf_min_ratio = 0.05;
    trf_max_iter = 15
    trf_max_iter_steihaug = 20;
    trf_tol_steihaug = args["steihaug-tolerance"];
    trf_tol_convergence = args["convergence-tolerance"];
    trf_init_scale_radius = 0.1;
    trf_save_every_iter = false;

    TRF_options = TrustRegionReflective.TRFOptions(
        trf_min_ratio,
        trf_max_iter,
        trf_max_iter_steihaug,
        trf_tol_steihaug,
<<<<<<< Updated upstream
        trf_tol_convergence,
=======
        1E-6,
>>>>>>> Stashed changes
        trf_init_scale_radius,
        trf_save_every_iter,
        false)

    x0 = T₁T₂ρˣρʸ(log(1.0), log(0.100),  1.0,  0.0) # note the logarithmic scaling to T1 and T2
    LB = T₁T₂ρˣρʸ(log(0.1), log(0.001), -Inf, -Inf) # note the logarithmic scaling to T1 and T2
    UB = T₁T₂ρˣρʸ(log(7.0), log(3.000),  Inf,  Inf) # note the logarithmic scaling to T1 and T2

    # Repeat x0, LB and UB for each voxel
    nr_voxels = Nx*Ny;

    # Check that there are no points with coil sensitivity zero within the mask
    @assert all( Cᵢ -> !iszero(sum(Cᵢ)), coil_sensitivities);

    qmaps = zeros(T₁T₂ρˣρʸ, Nx, Ny, nr_slices)

    slice_num = something(args["num-slices"], nr_slices)
    slice_start = something(args["start-slice"], (nr_slices - slice_num) ÷ 2 + 1)
    slice_end = slice_start + slice_num - 1

    slice_time_start = zeros(nr_slices)
    slice_time_end = zeros(nr_slices)

    time_total = @elapsed Threads.@threads :greedy for slice in slice_start:1:slice_end
        CompasToolkit.set_context(compas_context, Threads.threadid())
        slice_time_start[slice] = time()
        slice_time_end[slice] = time()

        pd_slice = pd[:,:,1,slice]
        mask_slice = findall(mask[:,:,slice])
        mask_len = length(mask_slice)

        if mask_len == 0
            println("Skipping slice $slice (of $nr_slices) as it is empty")
            continue
        end

        x0_slice = repeat(x0', mask_len) |> f32;
        x0_slice[:,3] .= real.(pd[mask_slice,1,slice])
        x0_slice[:,4] .= imag.(pd[mask_slice,1,slice])

        coordinates_slice = coordinates[mask_slice,1,slice]
        compas_data_slice = CompasToolkit.CompasArray(data[:,:,1,slice:slice])
        compas_coils = CompasToolkit.CompasArray(coil_sensitivities[mask_slice,:])

        thread_id = Threads.threadid()
        println("Thread $thread_id will process slice $slice (of $nr_slices) having $mask_len voxels")

        # Objective function
        objfun = (x, mode) -> objective(
                x, mode,
                compas_data_slice,
                compas_sequence,
                coordinates_slice,
                compas_coils,
                compas_trajectory, sequence)

        LB_slice = repeat(LB', mask_len) |> f32;
        UB_slice = repeat(UB', mask_len) |> f32;

        # Make plot function for further plotting of the iterations
        plotfun(it, state) = () #plot_T₁T₂ρ(optim_to_physical_pars(x), Nx, Ny, figtitle)
        #plotfun(it, state) = plot_T₁T₂ρ(optim_to_physical_pars(state.x[:,it]), Nx, Ny, "Iteration")
        #plotfun(x0_slice)

        to = TimerOutputs.TimerOutput()

        # Run non-linear solver
        time_slice = @elapsed output = TrustRegionReflective.trust_region_reflective(
                objfun, vec(x0_slice), vec(LB_slice), vec(UB_slice), plotfun, to, TRF_options,)

        q = optim_to_physical_pars(output)
        qmaps[mask_slice,slice] = q

        println("Thread $thread_id processed slice $slice of $nr_slices, took $time_slice seconds")
        slice_time_end[slice] = time()
    end

    println("Done. Took $time_total seconds")

    return qmaps, mask, (slice_time_start, slice_time_end)
end

args = parse_args()
output, mask, slice_time = main(args)
output_file = args["output"]

if !isempty(output_file)
    jldsave(
        output_file;
        mask=Array{Int8}(mask),
        T1=Array{Float32}(StructArray(output).T₁),
        T2=Array{Float32}(StructArray(output).T₂),
        rho_x=Array{Float32}(StructArray(output).ρˣ),
        rho_y=Array{Float32}(StructArray(output).ρʸ),
        slice_time_start=Array{Float64}(slice_time[1]),
        slice_time_end=Array{Float64}(slice_time[2]),
    )

    println("Wrote output to $output_file")
end

# Plot results:
# qmaps.T₁
# qmaps.T₂
# complex.(qmaps.ρˣ, qmaps.ρʸ)
