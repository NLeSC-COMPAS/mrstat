import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def compute_clim(data):
    return np.quantile(data, (.5, .95))

def print_stats(name, data, mask):
    data = np.ma.masked_array(data, mask=mask == 0)
    print("=" * 20 + " " + name + " " + "=" * 20)
    print(" - shape:", data.shape)
    print(" - min:", np.amin(data))
    print(" - max:", np.amax(data))
    print(" - mean:", np.average(data))
    print(" - stddev:", np.std(data))
    print(" - zeros:", np.average(data == 0))
    print(" - Q1:", np.nanquantile(data.filled(np.nan), .25))
    print(" - Q2:", np.nanquantile(data.filled(np.nan), .5))
    print(" - Q3:", np.nanquantile(data.filled(np.nan), .75))

def compute_error(output, reference):
    return np.abs(output - reference) / np.maximum(1e-5, np.abs(reference))

def main(output_file, reference_file):
    print(f"loading {output_file}")

    df = h5py.File(output_file)
    T1 = np.array(df["T1"])
    T2 = np.array(df["T2"])
    rho_x = np.array(df["rho_x"])
    rho_y = np.array(df["rho_y"])
    rho = np.sqrt(rho_x**2 + rho_y**2)
    mask = np.array(df["mask"])

    print_stats("T1", T1, mask)
    print_stats("T2", T2, mask)
    print_stats("rho", rho, mask)

    if reference_file:
        print(f"loading reference {output_file}")

        df_ref = h5py.File(reference_file)
        rho_x = np.array(df_ref["rho_x"])
        rho_y = np.array(df_ref["rho_y"])

        T1 = compute_error(T1, df_ref["T1"])
        T2 = compute_error(T2, df_ref["T2"])
        rho = compute_error(rho, np.sqrt(rho_x**2 + rho_y**2))

        print_stats("T1 (error)", (T1), mask)
        print_stats("T2 (error)", (T2), mask)
        print_stats("rho (error)", (rho), mask)

        mask -= df_ref["mask"]

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
    thread_ids = df.get("slice_thread_id", [0] * nslices)
    nthreads = np.amax(thread_ids) + 1

    times_end -= np.amin(times_start)
    times_start -= np.amin(times_start)
    time_total = np.amax(times_end)

    times_ax = plt.subplot(233)
    cmap = plt.get_cmap("tab10")
    plt.title("Timeline")
    plt.xlabel("Time (sec)")
    plt.ylabel("Slice id")

    for i in range(nslices):
        plt.plot(
                [times_start[i], times_end[i]],
                [i,i],
                c=cmap(thread_ids[i] / nthreads),
        )

    def update(val):
        sliceid = int(val)
        im_t1.set_data(T1[sliceid])
        im_t2.set_data(T2[sliceid])
        im_rho.set_data(rho[sliceid])
        im_mask.set_data(mask[sliceid])

        t = (times_start[sliceid] + times_end[sliceid]) / 2
        times_ax.set_xlim(t - .1 * time_total, t + .1 * time_total)

    ax = plt.subplot(236)
    slider = Slider(
            ax=ax,
            label="Slice",
            valmin=0,
            valmax=nslices-1,
            valinit=sliceid
    )
    slider.on_changed(update)

    title = output_file
    if reference_file:
        title += f" vs {reference_file}"

    plt.suptitle(title)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

if __name__ == "__main__":
    output_file = sys.argv[1]
    reference_file = sys.argv[2] if len(sys.argv) > 2 else None

    main(output_file, reference_file)
