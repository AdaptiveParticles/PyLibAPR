import pyapr
import numpy as np


def get_apr(image, rel_error=0.1, gradient_smoothing=2, verbose=True, params=None):

    # check that the image array is c-contiguous
    if not image.flags['C_CONTIGUOUS']:
        print('WARNING: \'image\' argument given to get_apr is not C-contiguous \n'
              'input image has been replaced with a C-contiguous copy of itself')
        image = np.ascontiguousarray(image)

    # Initialize objects
    apr = pyapr.APR()

    if params is None:
        par = pyapr.APRParameters()
        par.auto_parameters = True
        par.rel_error = rel_error
        par.gradient_smoothing = gradient_smoothing
    else:
        par = params

    if image.dtype == np.float32:
        parts = pyapr.FloatParticles()
        converter = pyapr.converter.FloatConverter()
    elif image.dtype == np.uint16:
        parts = pyapr.ShortParticles()
        converter = pyapr.converter.ShortConverter()
    # elif image.dtype in {'byte', 'uint8'}:  # currently not working
    #     parts = pyapr.ByteParticles()
    #     converter = pyapr.converter.ByteConverter()
    else:
        errstr = 'pyapr.converter.get_apr: input image dtype must be numpy.uint16 or numpy.float32, ' \
                 'but {} was given'.format(image.dtype)
        raise TypeError(errstr)

    converter.set_parameters(par)
    converter.verbose = verbose

    # Compute the APR and sample particles
    converter.get_apr(apr, image)
    parts.sample_image(apr, image)

    return apr, parts


def get_apr_interactive(image, rel_error=0.1, gradient_smoothing=2, verbose=True, params=None, slider_decimals=1):

    # check that the image array is c-contiguous
    if not image.flags['C_CONTIGUOUS']:
        print('WARNING: \'image\' argument given to get_apr_interactive is not C-contiguous \n'
              'input image has been replaced with a C-contiguous copy of itself')
        image = np.ascontiguousarray(image)

    while image.ndim < 3:
        image = np.expand_dims(image, axis=0)

    # Initialize objects
    io_int = pyapr.InteractiveIO()
    apr = pyapr.APR()

    if params is None:
        par = pyapr.APRParameters()
        par.auto_parameters = False
        par.rel_error = rel_error
        par.gradient_smoothing = gradient_smoothing
    else:
        par = params

    if image.dtype == np.float32:
        parts = pyapr.FloatParticles()
        converter = pyapr.converter.FloatConverter()
    elif image.dtype == np.uint16:
        parts = pyapr.ShortParticles()
        converter = pyapr.converter.ShortConverter()
    # elif image.dtype in {'byte', 'uint8'}:  # currently not working
    #     parts = pyapr.ByteParticles()
    #     converter = pyapr.converter.ByteConverter()
    else:
        errstr = 'pyapr.converter.get_apr_interactive: input image dtype must be numpy.uint16 or numpy.float32, ' \
                 'but {} was given'.format(image.dtype)
        raise TypeError(errstr)

    converter.set_parameters(par)
    converter.verbose = verbose

    # launch interactive APR converter
    io_int.interactive_apr(converter, apr, image, slider_decimals=slider_decimals)

    if verbose:
        print("Total number of particles: {}".format(apr.total_number_particles()))
        print("Number of pixels in original image: {}".format(image.size))
        cr = image.size/apr.total_number_particles()
        print("Compuational Ratio: {:7.2f}".format(cr))

    # sample particles
    parts.sample_image(apr, image)

    return apr, parts


def find_parameters_interactive(image, rel_error=0.1, gradient_smoothing=0, verbose=True, params=None, slider_decimals=1):

    # check that the image array is c-contiguous
    if not image.flags['C_CONTIGUOUS']:
        print('WARNING: \'image\' argument given to find_parameters_interactive is not C-contiguous \n'
              'input image has been replaced with a C-contiguous copy of itself')
        image = np.ascontiguousarray(image)

    while image.ndim < 3:
        image = np.expand_dims(image, axis=0)

    # Initialize objects
    io_int = pyapr.filegui.InteractiveIO()
    apr = pyapr.APR()
    if params is None:
        par = pyapr.APRParameters()
        par.auto_parameters = False
        par.rel_error = rel_error
        par.gradient_smoothing = gradient_smoothing
    else:
        par = params

    if image.dtype == np.float32:
        converter = pyapr.converter.FloatConverter()
    elif image.dtype == np.uint16:
        converter = pyapr.converter.ShortConverter()
    # elif image.dtype in {'byte', 'uint8'}:  # currently not working
    #     converter = pyapr.converter.ByteConverter()
    else:
        errstr = 'pyapr.converter.find_parameters_interactive: input image dtype must be numpy.uint16 or numpy.float32, ' \
                 'but {} was given'.format(image.dtype)
        raise TypeError(errstr)

    converter.set_parameters(par)
    converter.verbose = verbose

    # launch interactive APR converter
    par = io_int.find_parameters_interactive(converter, apr, image, slider_decimals=slider_decimals)

    if verbose:
        print("---------------------------------")
        print("Using the following parameters:")
        print("grad_th = {}, sigma_th = {}, Ip_th = {}".format(par.grad_th, par.sigma_th, par.Ip_th))
        print("---------------------------------")

    return par
