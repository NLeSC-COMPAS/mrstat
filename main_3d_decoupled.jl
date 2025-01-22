using Revise
using BlochSimulators
using StaticArrays
using LinearAlgebra
using Statistics
using StructArrays
using LinearMaps
using ImagePhantoms
using PythonPlot
using ComputationalResources
using CompasToolkit
using Random
using JLD2

include("TrustRegionReflective/TrustRegionReflective.jl")
include("DerivativeOperations/DerivativeOperations.jl")

using .TrustRegionReflective
using .DerivativeOperations

include("utils/make_phantom.jl")
include("utils/objective.jl")
include("utils/RelaxationColors.jl")
include("utils/pythonplot.jl")

# Load JLD2 file with data, sequence, trajectory and coordinates

    @load "mrstat_3d_decoupled.jld2" data sequence trajectory coordinates

# Don't ask

    # Make RF train twice as long
    RF_train = copy(sequence.RF_train)
    push!(sequence.RF_train, zero(RF_train)...)
    # insert zero after each RF pulse
    RF_train_padded = [transpose(RF_train) ; zero(transpose(RF_train))] |> vec 
    sequence.RF_train .= RF_train_padded
    
# Make coil sensitivities

    Nx = size(coordinates, 1)
    Ny = size(coordinates, 2)

    ncoils = 1
    coil_sensitivities = complex(ones(ncoils,Nx * Ny))
    coil_sensitivities = map(SVector{ncoils}, eachcol(coil_sensitivities))

# Compas

    compas_context = CompasToolkit.init_context(0)

    compas_sequence = CompasToolkit.FispSequence(
        sequence.RF_train, 
        sequence.sliceprofiles, 
        sequence.TR,
        sequence.TE,
        sequence.max_state,
        sequence.TI
    )

    compas_trajectory = CompasToolkit.CartesianTrajectory(
        trajectory.nreadouts, 
        trajectory.nsamplesperreadout, 
        trajectory.Δt, 
        trajectory.k_start_readout, 
        trajectory.Δk_adc
    )

    compas_coils = CompasToolkit.make_array(compas_context, ComplexF32.(vec(only.(coil_sensitivities))))


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

    for slice in 30:nr_slices-30 # First and last few slices aren't that interesting


        compas_data_current_slice = # TODO: SOMETHING(data[:,:,slice])
        coordinates_current_slice = vec(coordinates[:,:,1,slice])

        # Make plot function for further plotting of the iterations
        objfun = (x,mode) -> objective(x, resource, mode, compas_data, compas_sequence, coordinates, compas_coils, compas_trajectory)

        # Run Trust Refion Reflective solver
        trf_min_ratio = 0.05;
        trf_max_iter = 15
        trf_max_iter_steihaug = 20;
        trf_tol_steihaug = 0.1;
        trf_init_scale_radius = 0.1;
        trf_save_every_iter = false;

        TRF_options = TrustRegionReflective.SolverOptions(
            trf_min_ratio,
            trf_max_iter,
            trf_max_iter_steihaug,
            trf_tol_steihaug,
            trf_init_scale_radius,
            trf_save_every_iter)

        # plotfun(x, figtitle) = plot_T₁T₂ρ(optim_to_physical_pars(x), N, N, figtitle)

        # plotfun(x0, "Initial Guess")

        # Run non-linear solver

        output = TrustRegionReflective.solver(objfun, vec(x0), vec(LB), vec(UB), TRF_options, plotfun)

        q = optim_to_physical_pars(output.x[:,end])
        qmaps[:,:,slice] = reshape(q, Nx, Ny)
    end

# Plot results:
# qmaps.T₁
# qmaps.T₂
# complex.(qmaps.ρˣ, qmaps.ρʸ)
#   