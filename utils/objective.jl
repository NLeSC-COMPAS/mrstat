using BlochSimulators
using Statistics
using PythonPlot


function objective(optimpars::Vector{<:Real}, mode, raw_data, sequence, coordinates, coil_sensitivities, trajectory)
    #GC.gc(true)

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
    magnetization = magnetization_original #repeat(collect(magnetization_original), inner=(1, 2))

    # Apply phase encoding
    echos = CompasToolkit.phase_encoding(
        magnetization,
        parameters,
        trajectory
    )

    # Compute signal
    s = CompasToolkit.magnetization_to_signal(echos, parameters, trajectory, coil_sensitivities)

    # Compute residual r
    f, r = CompasToolkit.compute_residual(s, raw_data)

    ncoils = size(r, 3)
    r_host = reshape(collect(r), :, ncoils)
    r_host = map(SVector{ncoils}, eachrow(r_host))

    # Compute cost f
    f = 0.5f0 * f

    if mode == "fr"
        return f, r_host
    end

    # Compute partial derivatives of magnetization at echo time
    ∂echos = CompasToolkit.simulate_magnetization_derivatives(magnetization_original, parameters, sequence)

    # Apply phase encoding
    ∂echos = CompasToolkit.phase_encoding(∂echos, parameters, trajectory)

    # Compute gradient
    g = CompasToolkit.compute_jacobian_hermitian(
        echos,
        ∂echos,
        parameters,
        trajectory,
        coil_sensitivities,
        r
    )

    # Reshape as vector of reals
    g = collect(g)
    g = reshape(g, :)
    g = real.(g)

    if mode == "frg"
        return f, r_host, g
    end

    # Make Gauss-Newton matrix multiply function
    reJᴴJ(x) = begin
        np = 4 # nr of reconstruction parameters per voxel
        x = reshape(x,:,np)
        x = ComplexF32.(x)

        y = CompasToolkit.compute_jacobian(
            echos,
            ∂echos,
            parameters,
            trajectory,
            coil_sensitivities,
            x
        )

        z = CompasToolkit.compute_jacobian_hermitian(
            echos,
            ∂echos,
            parameters,
            trajectory,
            coil_sensitivities,
            y
        )

        z = collect(z)
        return real.(reshape(z, :))
    end

    H = LinearMap(
        v -> reJᴴJ(v),
        v -> v, # adjoint operation not used
    length(g),length(g));

    return f, r_host, g, H
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
