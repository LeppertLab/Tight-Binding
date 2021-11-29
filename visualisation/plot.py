import numpy as np


def plot_band_structure(ax, n_pos_symmetry_points, band_structure, symmetry_points=None):
    ax.set_ylabel("Energy (eV)")

    if symmetry_points is not None:
        ax.set_xticks(n_pos_symmetry_points)
        ax.set_xticklabels([f"${symmetry_point}$" for symmetry_point in symmetry_points])

    ax.grid(which='major', axis='x', linewidth='0.5', color='lightgrey')

    ax.axhline(color='lightgrey', linewidth=0.5, linestyle="--")

    ax.set_xlim([n_pos_symmetry_points[0], n_pos_symmetry_points[-1]])

    x = np.linspace(0, np.shape(band_structure)[1], np.shape(band_structure)[1])
    for band in band_structure:
        ax.plot(x, band)


def visualize_fit(fig, n_pos_symmetry_points, original_eigenvalues, fitted_eigenvalues, symmetry_points):
    if np.shape(original_eigenvalues) == np.shape(fitted_eigenvalues):
        error = fitted_eigenvalues - original_eigenvalues
        print('f = ' + str(np.average(np.abs(np.power(error, 2)))))
        ax1 = fig.add_subplot(221)
        ax1.set_title("Fitted bandstructure")
        ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
        ax2.set_title("Input bandstructure")
        ax3 = fig.add_subplot(223)
        ax3.set_title("Error")
        ax4 = fig.add_subplot(224)
        ax4.set_title("Histogram Error")
        plot_band_structure(ax1, n_pos_symmetry_points, fitted_eigenvalues, symmetry_points)
        plot_band_structure(ax2, n_pos_symmetry_points, original_eigenvalues, symmetry_points)
        plot_band_structure(ax3, n_pos_symmetry_points, error, symmetry_points)
        ax4.set_ylabel("Energy (eV)")
        ax4.hist(np.ravel(error), bins=50, edgecolor='black')
        ax4.set_xlabel("Energy (eV)")
        ax4.set_ylabel("Number of points")
    else:
        print('Shape of the bandstructure and the external bandstructure do not match. Skipping the error plots.')
        ax1 = fig.add_subplot(211)
        ax1.set_title("Input bandstructure")
        ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
        ax2.set_title("Fitted bandstructure")
        plot_band_structure(ax1, n_pos_symmetry_points, original_eigenvalues, symmetry_points)
        plot_band_structure(ax2, n_pos_symmetry_points, fitted_eigenvalues, symmetry_points)
