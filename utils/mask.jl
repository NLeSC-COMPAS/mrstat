import Pkg; Pkg.add("ImageFiltering")
using ImageFiltering

"""
    _update_mask_for_location!(ctx, relative_threshold, location::Union{Int,Colon})

Update the mask based on thresholding of the proton density (magnitude). The threshold is calculated `relative_threshold` using the `_calculate_threshold` function.

# Note
- If `location` is an integer, the mask is updated for that location.
- If `location` is a colon, the mask is updated for all locations.
- The voxels for which the mask is true are considered the "active voxels" for which tissue properties are estimated in the non-linear reconstruction.
- The `collect` and `parent` is needed for performance reasons (`NamedDimsArray`s have some issues on GPU).
"""
function calculate_ρ_mask(ρ, relative_threshold::Real=0.3)

    # Extract the absolute values of the proton density
    ρ_abs = collect(abs.(ρ))

    # Smooth the proton density prior to thresholding to reduce effects of noise/outliers
    ρ_abs = _smooth_proton_density(ρ_abs)

    # # Calculate the threshold
    threshold = _calculate_threshold(ρ_abs, relative_threshold)

    # Generate mask based on proton density threshold
    mask = ρ_abs .> threshold

    return mask
end

"""
    _calculate_threshold(ρ_abs, relative_threshold)

Calculate the threshold for the proton density based on the relative threshold and the peak magnitude.

# Note
- The `quantile` function expects the proton density (magnitude) as vector.
"""
function _calculate_threshold(ρ_abs, relative_threshold::Real)
    # return quantile(vec(ρ_abs), relative_threshold)
    return relative_threshold * maximum(vec(ρ_abs))
end

"""
    _smooth_proton_density(ρ_abs)

Smooths the proton density image using a Gaussian filter.

# Returns
- `ρ_abs_smoothed`: The smoothed proton density image.
"""
function _smooth_proton_density(ρ_abs)

    # TODO: Make the filtersize an option (relative at least)
    if ndims(ρ_abs) == 3
        nx, ny, nz = size(ρ_abs)
        filter_size = (max(nx / 32, 1), max(ny / 32, 1), max(nz / 32, 1))
    elseif ndims(ρ_abs) == 4
        nx, ny, nz, nloc = size(ρ_abs)
        filter_size = (max(nx / 32, 1), max(ny / 32, 1), max(nz / 32, 1), max(nloc / 6, 1))
    else
        error("Proton density must be 3D or 4D")
    end

    ρ_abs_smoothed = imfilter(ρ_abs, Kernel.gaussian(filter_size))

    return ρ_abs_smoothed
end
