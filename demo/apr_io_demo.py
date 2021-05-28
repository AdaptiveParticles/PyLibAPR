import pyapr
import numpy as np
from skimage import io as skio


def main():
    """
    This demo converts a selected TIFF image to an APR, writes the result to file and then reads the file.
    """

    io_int = pyapr.filegui.InteractiveIO()

    # Read in an image
    fpath = io_int.get_tiff_file_name()
    img = skio.imread(fpath).astype(np.uint16)

    # Instantiate objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()
    par = pyapr.APRParameters()
    converter = pyapr.converter.ShortConverter()

    # Set some parameters
    par.auto_parameters = False
    par.rel_error = 0.1
    par.Ip_th = 10
    par.grad_th = 1
    par.gradient_smoothing = 2
    par.sigma_th = 20
    par.sigma_th_max = 10
    converter.set_parameters(par)
    converter.verbose = False

    # Compute APR and sample particle values
    converter.get_apr(apr, img)
    parts.sample_image(apr, img)

    # Compute and display the computational ratio
    numParts = apr.total_number_particles()
    numPix = img.size
    cr = numPix / numParts
    print('Input image size: {} pixels, APR size: {} particles --> Computational Ratio: {}'.format(numPix, numParts, cr))

    # Save the APR to file
    fpath_apr = io_int.save_apr_file_name()  # get save path from gui
    pyapr.io.write(fpath_apr, apr, parts)    # write apr and particles to file

    # Read the newly written file
    apr2 = pyapr.APR()
    parts2 = pyapr.ShortParticles()
    pyapr.io.read(fpath_apr, apr2, parts2)

    # check that particles are equal at a single, random index
    ri = np.random.randint(0, numParts-1)
    assert parts[ri] == parts2[ri]

    # check some APR properties
    assert apr.total_number_particles() == apr2.total_number_particles()
    assert apr.level_max() == apr2.level_max()
    assert apr.level_min() == apr2.level_min()


if __name__ == '__main__':
    main()
