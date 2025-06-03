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

GC.enable_logging(true)

include("DerivativeOperations/DerivativeOperations.jl")
using .DerivativeOperations

include("utils/make_phantom.jl")
include("utils/objective.jl")
include("utils/RelaxationColors.jl")
include("utils/pythonplot.jl")

# Load JLD2 file with data, sequence, trajectory and coordinates

    @load "mrstat_3d_decoupled_with_pd2.jld2" data sequence trajectory coordinates pd

# Fix k0
    kˣ = -real.(trajectory.Δk_adc) * trajectory.nsamplesperreadout / 2
    kʸ = imag.(trajectory.k_start_readout)
    trajectory.k_start_readout .= kˣ .+ kʸ .* im

# Don't ask

    # Make RF train twice as long
    #RF_train = copy(sequence.RF_train)
    #push!(sequence.RF_train, zero(RF_train)...)
    # insert zero after each RF pulse
    #RF_train_padded = [transpose(RF_train) ; zero(transpose(RF_train))] |> vec 
    #sequence.RF_train .= RF_train_padded

    sliceprofiles = ones(length(sequence.RF_train),1) .|> complex;
    unwrap(::Val{x}) where x = x
    
# Make coil sensitivities

    Nx = size(coordinates, 1)
    Ny = size(coordinates, 2)

    ncoils = 1
    coil_sensitivities = ComplexF32.(ones(Nx * Ny, ncoils))
    #coil_sensitivities = map(SVector{ncoils}, eachcol(coil_sensitivities))
    #coil_sensitivities = ComplexF32.(vec(only.(coil_sensitivities)))

# Compas

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

    bloch = (
        sequence=sequence,
        trajectory=trajectory
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

    nr_slices = size(coordinates, 4)

    qmaps = zeros(T₁T₂ρˣρʸ, Nx, Ny, nr_slices)

    #for slice in 30:nr_slices-30 # First and last few slices aren't that interesting
    #for slice in fld(nr_slices, 2)-5:fld(nr_slices, 2)+5
    time = @elapsed Threads.@threads :dynamic for slice in 100:1:115
        CompasToolkit.set_context(compas_context)

        thread_id = Threads.threadid()
        println("Thread $thread_id will process slice $slice of $nr_slices")

        compas_data_slice = data[:,:,1,slice:slice]
        coordinates_slice = vec(coordinates[:,:,1,slice])

        x0_slice = copy(x0)
        x0_slice[:,3] .= real.(vec(pd[:,:,1,slice]))
        x0_slice[:,4] .= imag.(vec(pd[:,:,1,slice]))

        # Make plot function for further plotting of the iterations
        objfun = (x,mode) -> objective(x, CUDALibs(), mode, compas_data_slice, compas_sequence, coordinates_slice, compas_coils, compas_trajectory, bloch)

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

        plotfun(x, figtitle) = () #plot_T₁T₂ρ(optim_to_physical_pars(x), Nx, Ny, figtitle)
        plotfun(x0_slice, "Initial Guess")

        to = TimerOutputs.TimerOutput()

        # Run non-linear solver
        time = @elapsed output = TrustRegionReflective.trust_region_reflective(
            objfun, vec(x0_slice), vec(LB), vec(UB), plotfun, to, TRF_options)

        q = optim_to_physical_pars(output)
        qmaps[:,:,slice] = reshape(q, Nx, Ny)
        #plot_T₁T₂ρ(q, Nx, Ny, "Result slice $slice")

        println("Thread $thread_id processed slice $slice of $nr_slices, took $time seconds")
    end

    println("Done. Took $time seconds")

# Plot results:
# qmaps.T₁
# qmaps.T₂
# complex.(qmaps.ρˣ, qmaps.ρʸ)
#
