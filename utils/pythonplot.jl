# make colormaps
loLevT₁ = 0.0; upLevT₁ = 2.5;
loLevT₂ = 0.0; upLevT₂ = 0.35;
img = 0.1*ones(10,10)
_, rgb_vec_T₁ = relaxationColorMap("T1", img, loLevT₁, upLevT₁)
_, rgb_vec_T₂ = relaxationColorMap("T2", img, loLevT₂, upLevT₂)
lipari = PythonPlot.ColorMap("lipari", rgb_vec_T₁, length(rgb_vec_T₁), 1.0)
navia  = PythonPlot.ColorMap("navia",  rgb_vec_T₂, length(rgb_vec_T₂), 1.0)

plot_timestep = Threads.Atomic{Int}(0)

function plot_T₁T₂ρ(x::AbstractArray{<:AbstractTissueProperties}, Nx, Ny, figtitle="")
    global plot_timestep
    q = StructArray(reshape(x,Nx,Ny))

    figure()

    subplot(221)
        imshow(q.T₁, clim=(0.0,2.5), cmap=lipari)
        colorbar()
        xlabel("T₁ [s]")
    subplot(222)
        imshow(q.T₂, clim=(0.0,0.35), cmap=navia)
        colorbar()
        xlabel("T₂ [s]")
    subplot(223)
        imshow(abs.(complex.(q.ρˣ, q.ρʸ)), clim=(0.0,2.0), cmap="gray")
        colorbar()
        xlabel("ρ [a.u.]")
    subplot(224)
        imshow(angle.(complex.(q.ρˣ, q.ρʸ)), cmap="hsv")
        colorbar()
        xlabel("ρ [angle]")

    suptitle(figtitle)

    index = Threads.atomic_add!(plot_timestep, 1)
    println("writing image to `image_$(index).pdf`")
    savefig("image_$(index).pdf")

end
