using BlochSimulators
using Statistics
using PythonPlot

function check_nans(arr)
    # Find all indices (CartesianIndices for N-D arrays) where `arr` is NaN.
    nan_positions = findall(isnan, collect(arr))
    
    for pos in nan_positions
        error("NaN found at index ", pos)
        exit()
    end
end

function plot_file(name, mat)
    figure()
    imshow(abs.(collect(mat)), aspect="auto")
    colorbar()
    savefig(string(name, ".pdf"))
end

function objective(optimpars::Vector{<:Real}, resource, mode, raw_data, sequence, coordinates, coil_sensitivities, trajectory, bloch)
    GC.gc(true)

    # We compute the residual rᵢ = ||d Σᵢ (dᵢ - M(T₁,T₂,B₁,B₀)*Cᵢ*ρ)
    # f = (1/2) * |r|^2
    # The gradient is computed as g = ℜ(Jᴴr)

    # mode 0 -> compute f and r only
    # mode 1 -> compute f, r and g
    # mode 2 -> compute f, r, g and assemble approximate Hessian

    # Convert optimpars (Vector{<:Real}) to Vector{<:AbstractTissueParameters} to be used in simulations
    parameters = optim_to_physical_pars(optimpars, coordinates)

    # Send to gpu device
    nvoxels = length(parameters)
    parameters = CompasToolkit.TissueParameters(
        nvoxels,
        StructArray(parameters).T₁,
        StructArray(parameters).T₂,
        fill(1, nvoxels), # B1
        fill(0, nvoxels), # B0
        StructArray(parameters).ρˣ,
        StructArray(parameters).ρʸ,
        collect(coordinates.x),
        collect(coordinates.y)
    )


    # Compute magnetization at echo times
    magnetization_original = CompasToolkit.simulate_magnetization(parameters, sequence)
    check_nans(magnetization_original)
    magnetization = repeat(collect(magnetization_original), inner=(1, 2))
    check_nans(magnetization)
    plot_file("magnetization", magnetization)

    # Apply phase encoding
    echos = CompasToolkit.phase_encoding(
        magnetization,
        parameters,
        trajectory
    )
    check_nans(echos)

    # Compute signal
    s = CompasToolkit.magnetization_to_signal(echos, parameters, trajectory, coil_sensitivities)
    check_nans(s)

    # Compute residual r
    f, r = CompasToolkit.compute_residual(s, raw_data)

    ncoils = size(r, 3)
    r_host = reshape(collect(r), :, ncoils)
    r_host = map(SVector{ncoils}, eachrow(r_host))

    # Compute cost f
    f = 0.5 * f

    #BlochSimulators
    bloch_parameters = optim_to_physical_pars(optimpars, coordinates)
    slice_profiles = ones(length(bloch.sequence.RF_train),1) .|> complex
    bloch_sequence = FISP2D(
        collect(bloch.sequence.RF_train),
        collect(slice_profiles),
        bloch.sequence.TR,
        bloch.sequence.TE,
        bloch.sequence.max_state,
        bloch.sequence.TI
    )

    bloch_echos = simulate_magnetization(CPU1(), bloch_sequence, bloch_parameters)
    bloch_echos = repeat(collect(bloch_echos), inner=(2, 1))

    println("max error mag: ", maximum(abs.((collect(bloch_echos)) .- (transpose(collect(magnetization))))))
    println("mean mag: ", mean(abs.(collect(bloch_echos))))

    bloch_trajectory = CartesianTrajectory2D(
        bloch.trajectory.nreadouts,
        bloch.trajectory.nsamplesperreadout,
        bloch.trajectory.Δt,
        collect(bloch.trajectory.k_start_readout),
        real(bloch.trajectory.Δk_adc),
        [1,2,3], #trajectory.py,
        Int64(1) #trajectory.readout_oversampling
    )

    phase_encoding!(bloch_echos, bloch_trajectory, coordinates)


    println("max error echos: ", maximum(abs.((collect(bloch_echos)) .- (transpose(collect(echos))))))
    println("mean echos: ", mean(abs.(collect(bloch_echos))))
    bloch_echos = gpu(transpose(collect(echos)))

    println("bloch_echos: ", summary(bloch_echos))
    println("bloch_parameters: ", summary(bloch_parameters))
    println("coordinates: ", summary(coordinates))
    println("coil_sensitivities: ", summary(coil_sensitivities))

    #jldsave("dump_3d_decoupled.jld2"; 
    #    echos=collect(bloch_echos), 
    #    parameters=bloch_parameters, 
    #    trajectory=bloch_trajectory,
    #    coordinates=coordinates,
    #    coil_sensitivities=collect(coil_sensitivities))

    bloch_signal = magnetization_to_signal(
        CUDALibs(),
        gpu(bloch_echos),
        #bloch_sequence,
        gpu(bloch_parameters),
        gpu(bloch_trajectory),
        gpu(coordinates),
        gpu(collect(coil_sensitivities))
    )
    bloch_signal = reshape(collect(bloch_signal), trajectory.samples_per_readout, trajectory.nreadouts)

    # println("bloch_signal: ", summary(bloch_signal))
    # println("s: ", summary(s))

    # println("bloch_signal: ", collect(bloch_signal)[1:3,1:3])
    # println("s: ", collect(s)[1:3,1:3,1])

    #println("max error signal: ", maximum(abs.(vec(collect(bloch_signal)) .- vec(transpose(collect(s)[:,:,1])))))
    #println("mean signal: ", mean(abs.(collect(bloch_signal))))

    println("max error signal: ", maximum(abs.(vec(collect(bloch_signal)) .- vec((collect(s)[:,:,1])))))
    println("mean signal: ", mean(abs.(collect(bloch_signal))))

    s = collect(s)
    s[:,:,1] = bloch_signal 
    plot_file("signal", s)

    # Compute residual r
    f, r = CompasToolkit.compute_residual(s, raw_data)
    check_nans(r)
    bloch_r = bloch_signal .- raw_data

    println("r: ", summary(r))
    println("bloch_r: ", summary(bloch_r))
    println("raw_data: ", summary(raw_data))

    println("max error residual: ", maximum(abs.((collect(bloch_r)) .- ((collect(r))))))
    println("mean residual: ", mean(abs.(collect(bloch_r))))

    #exit()

    if mode == 0

        return f, r_host

    elseif mode > 0

        # Compute partial derivatives of magnetization at echo time
        ∂echos = CompasToolkit.simulate_magnetization_derivatives(magnetization_original, parameters, sequence)

        check_nans(∂echos.T1)
        check_nans(∂echos.T2)
        plot_file("derivative_T1", ∂echos.T1)
        plot_file("derivative_T2", ∂echos.T2)

        ∂echos = (
            T1=repeat(collect(∂echos.T1), inner=(1, 2)),
            T2=repeat(collect(∂echos.T2), inner=(1, 2))
        )

        # Apply phase encoding
        ∂echos = (
            T1=CompasToolkit.phase_encoding(∂echos.T1, parameters, trajectory),
            T2=CompasToolkit.phase_encoding(∂echos.T2, parameters, trajectory)
        )
        check_nans(∂echos.T1)
        check_nans(∂echos.T2)

        # Compute gradient
        g = CompasToolkit.compute_jacobian_hermitian(
            echos,
            ∂echos,
            parameters,
            trajectory,
            coil_sensitivities,
            r
        )
        check_nans(g)

        # Reshape as vector of reals
        g = collect(g)
        g = reshape(g, :)
        g = real.(g)

        mode == 1 && return f, r_host, g

        # Make Gauss-Newton matrix multiply function
        reJᴴJ(x) = begin
            np = 4 # nr of reconstruction parameters per voxel
            x = reshape(x,:,np)
            x = ComplexF32.(x)
            check_nans(x)
            
            y = CompasToolkit.compute_jacobian(
                echos,
                ∂echos,
                parameters,
                trajectory,
                coil_sensitivities,
                x
            )
            check_nans(y)
            
            z = CompasToolkit.compute_jacobian_hermitian(
                echos,
                ∂echos,
                parameters,
                trajectory,
                coil_sensitivities,
                y
            )
            check_nans(z)

            z = collect(z)
            return real.(reshape(z, :))
        end

        H = LinearMap(
            v -> reJᴴJ(v),
            v -> v, # adjoint operation not used
        length(g),length(g));

        return f, r_host, g, H
    end
end

function optim_to_physical_pars(optimpars)

    optimpars = reshape(optimpars,:,4)
    T₁ = exp.(optimpars[:,1])
    T₂ = exp.(optimpars[:,2])
    ρˣ = optimpars[:,3]
    ρʸ = optimpars[:,4]

    return map(T₁T₂ρˣρʸ, T₁, T₂, ρˣ, ρʸ)
end

function optim_to_physical_pars(optimpars, coordinates)

    optimpars = reshape(optimpars,:,4)
    T₁ = exp.(optimpars[:,1])
    T₂ = exp.(optimpars[:,2])
    ρˣ = optimpars[:,3]
    ρʸ = optimpars[:,4]
    #x = collect(coordinates)
    #y = collect(coordinates)

    return map(T₁T₂ρˣρʸ, T₁, T₂, ρˣ, ρʸ)
end
