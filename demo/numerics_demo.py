import pyapr


def main():
    """
    This demo showcases some of the available numerics functionality on a selected APR
    """

    io_int = pyapr.filegui.InteractiveIO()
    fpath_apr = io_int.get_apr_file_name()      # get APR file path from gui

    # Instantiate APR and particle objects
    apr = pyapr.APR()
    parts = pyapr.ShortParticles()              # input particles can be float32 or uint16
    # parts = pyapr.FloatParticles()

    # Read from APR file
    pyapr.io.read(fpath_apr, apr, parts)

    output = pyapr.FloatParticles()

    # Compute gradient along a dimension (Sobel filter). dimension can be 0, 1 or 2
    pyapr.numerics.gradient(apr, parts, output, dimension=0, delta=1.0, sobel=True)
    pyapr.viewer.parts_viewer(apr, output)   # Display the result

    # Compute gradient magnitude (central finite differences)
    pyapr.numerics.gradient_magnitude(apr, parts, output, deltas=(1.0, 1.0, 1.0), sobel=False)
    pyapr.viewer.parts_viewer(apr, output)  # Display the result

    # Compute local standard deviation around each particle
    pyapr.numerics.local_std(apr, parts, output, size=(5, 5, 5))
    pyapr.viewer.parts_viewer(apr, output)  # Display the result


if __name__ == '__main__':
    main()
