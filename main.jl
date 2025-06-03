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
        "--file-name", "-f"
            help = "the input file"
            arg_type = String
            default = "mrstat_3d_decoupled_with_pd2.jld2"
        "--gc-debug"
            help = "enable garbage collection logging"
            action = :store_true
    end

    return ArgParse.parse_args(s)
end


function main(args)
    if args["gc-debug"]
        GC.enable_logging(true)
    end

# Load JLD2 file with data, sequence, trajectory and coordinates
    file_name = args["file-name"]
    @load file_name data sequence trajectory coordinates pd

    Nx = size(coordinates, 1)
    Ny = size(coordinates, 2)
    nr_slices = size(coordinates, 4)
    ncoils = 1

    println("parsing ", file_name)
    println("- num. slices: ", nr_slices)
    println("- num. readouts: ", trajectory.nreadouts)
    println("- num. samples per readout: ", trajectory.nsamplesperreadout)
    println("- num. voxels: ", Nx, "x", Ny)
    println("- num. coils: ", ncoils)

# Fix k0
    kˣ = -real.(trajectory.Δk_adc) * trajectory.nsamplesperreadout / 2
    kʸ = imag.(trajectory.k_start_readout)
    trajectory.k_start_readout .= kˣ .+ kʸ .* im

    sliceprofiles = ones(length(sequence.RF_train),1) .|> complex;
    unwrap(::Val{x}) where x = x

# Make coil sensitivities
    coil_sensitivities = ComplexF32.(ones(Nx * Ny, ncoils))

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

    compas_coils = CompasToolkit.CompasArray(coil_sensitivities)


# Add noise?

# Set reconstruction options

    x0 = T₁T₂ρˣρʸ(log(1.0), log(0.100),  1.0,  0.0) # note the logarithmic scaling to T1 and T2
    LB = T₁T₂ρˣρʸ(log(0.1), log(0.001), -Inf, -Inf) # note the logarithmic scaling to T1 and T2
    UB = T₁T₂ρˣρʸ(log(7.0), log(3.000),  Inf,  Inf) # note the logarithmic scaling to T1 and T2

    # Repeat x0, LB and UB for each voxel
    nr_voxels = Nx*Ny;

    x0 = repeat(x0', nr_voxels) |> f32;
    LB = repeat(LB', nr_voxels) |> f32;
    UB = repeat(UB', nr_voxels) |> f32;

    # Check that there are no points with coil sensitivity zero within the mask
    @assert all( Cᵢ -> !iszero(sum(Cᵢ)), coil_sensitivities);

    qmaps = zeros(T₁T₂ρˣρʸ, Nx, Ny, nr_slices)

    slice_num = something(args["num-slices"], nr_slices)
    slice_start = something(args["start-slice"], (nr_slices - slice_num) ÷ 2 + 1)
    slice_end = slice_start + slice_num - 1

    time = @elapsed Threads.@threads :dynamic for slice in slice_start:1:slice_end
        CompasToolkit.set_context(compas_context)

        thread_id = Threads.threadid()
        println("Thread $thread_id will process slice $slice of $nr_slices")

        compas_data_slice = data[:,:,1,slice:slice]
        coordinates_slice = vec(coordinates[:,:,1,slice])

        x0_slice = copy(x0)
        x0_slice[:,3] .= real.(vec(pd[:,:,1,slice]))
        x0_slice[:,4] .= imag.(vec(pd[:,:,1,slice]))

        # Run Trust Refion Reflective solver
        trf_min_ratio = 0.05;
        trf_max_iter = 5
        trf_max_iter_steihaug = 20;
        trf_tol_steihaug = 0.1;
        trf_init_scale_radius = 0.1;
        trf_save_every_iter = false;

        TRF_options = TrustRegionReflective.TRFOptions(
            trf_min_ratio,
            trf_max_iter,
            trf_max_iter_steihaug,
            trf_tol_steihaug,
            trf_init_scale_radius,
            trf_save_every_iter,
            false)

        # Objective function
        objfun = (x, mode) -> objective(
                x, mode,
                compas_data_slice,
                compas_sequence,
                coordinates_slice,
                compas_coils,
                compas_trajectory)


        # Make plot function for further plotting of the iterations
        plotfun(it, state) = () #plot_T₁T₂ρ(optim_to_physical_pars(x), Nx, Ny, figtitle)
        plotfun(it, state) = plot_T₁T₂ρ(optim_to_physical_pars(state.x[:,it]), Nx, Ny, "Iteration")
        #plotfun(x0_slice)

        to = TimerOutputs.TimerOutput()

        # Run non-linear solver
        time = @elapsed output = TrustRegionReflective.trust_region_reflective(
            objfun, vec(x0_slice), vec(LB), vec(UB), plotfun, to, TRF_options)

        q = optim_to_physical_pars(output)
        qmaps[:,:,slice] = reshape(q, Nx, Ny)

        println("Thread $thread_id processed slice $slice of $nr_slices, took $time seconds")
    end

    println("Done. Took $time seconds")

    return qmaps
end

qmap = main(parse_args())

# Plot results:
# qmaps.T₁
# qmaps.T₂
# complex.(qmaps.ρˣ, qmaps.ρʸ)
