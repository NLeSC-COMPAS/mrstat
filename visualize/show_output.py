import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def compute_clim(data):
    print(np.quantile(data, [.5, .10, .25, .50, .75, .90, .95]))
    return (0, np.quantile(data, .95))

def main(arg):
    df = h5py.File(arg)
    T1 = np.array(df["T1"])
    T2 = np.array(df["T2"])
    rho_x = np.array(df["rho_x"])
    rho_y = np.array(df["rho_y"])
    rho = np.sqrt(rho_x**2 + rho_y**2)
    mask = df["mask"]

    nslices,nx,ny = T1.shape
    sliceid = 0

    plt.subplot(231)
    plt.title("T1")
    im_t1 = plt.imshow(T1[sliceid], cmap="Reds", clim=compute_clim(T1))
    plt.colorbar()

    plt.subplot(232)
    plt.title("T2")
    im_t2 = plt.imshow(T2[sliceid], cmap="Blues", clim=compute_clim(T2))
    plt.colorbar()

    plt.subplot(234)
    plt.title("rho")
    im_rho = plt.imshow(rho[sliceid], cmap="viridis", clim=compute_clim(rho))
    plt.colorbar()

    plt.subplot(235)
    plt.title("mask")
    im_mask = plt.imshow(mask[sliceid], cmap="Greys_r", clim=compute_clim(mask))


    times_start = list(df["slice_time_start"])
    times_end = list(df["slice_time_end"])
    times_end -= np.amin(times_start)
    times_start -= np.amin(times_start)

    plt.subplot(233)
    plt.title("Timeline")
    plt.xlabel("Time (sec)")
    plt.ylabel("Slice id")
    plt.plot(
            [times_start, times_end],
            [np.arange(nslices)] * 2
    )

    def update(val):
        sliceid = int(val)
        im_t1.set_data(T1[sliceid])
        im_t2.set_data(T2[sliceid])
        im_rho.set_data(rho[sliceid])
        im_mask.set_data(mask[sliceid])

    ax = plt.subplot(236)
    slider = Slider(
            ax=ax,
            label="Slice",
            valmin=0,
            valmax=nslices-1,
            valinit=sliceid
    )
    slider.on_changed(update)

    plt.tight_layout(pad=0)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])
