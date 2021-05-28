import unittest
import pyapr
from skimage import io as skio
import numpy as np
import os


def constant_upsample(img, output_shape, factor=2):
    output = np.zeros(shape=output_shape, dtype=img.dtype)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            for k in range(output_shape[2]):
                output[i, j, k] = img[i//factor, j//factor, k//factor]
    return output


def expand(img):
    while img.ndim < 3:
        img = np.expand_dims(img, axis=0)
    return img


class BasicTests(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_files')
        self.impath_3D = os.path.abspath(os.path.join(self.data_dir, 'sphere_original.tif'))
        self.impath_2D = os.path.abspath(os.path.join(self.data_dir, 'sphere_2D.tif'))
        self.impath_1D = os.path.abspath(os.path.join(self.data_dir, 'sphere_1D.tif'))

        self.apr_parameters = pyapr.APRParameters()
        self.apr_parameters.sigma_th = 75
        self.apr_parameters.grad_th = 40
        self.apr_parameters.Ip_th = 0
        self.apr_parameters.gradient_smoothing = 0
        self.apr_parameters.output_dir = self.data_dir + '/'

    def __convert_image(self, impath, dtype, rel_err=0.1):
        self.apr_parameters.rel_error = rel_err
        self.apr_parameters.output_steps = True
        img = expand(skio.imread(impath).astype(dtype))
        apr, parts = pyapr.converter.get_apr(img, verbose=False, params=self.apr_parameters)

        # read pipeline steps
        lis = expand(skio.imread(os.path.join(self.data_dir, 'local_intensity_scale_rescaled.tif')).astype(np.float32))
        grad = expand(skio.imread(os.path.join(self.data_dir, 'gradient_step.tif')).astype(np.float32))

        # upsample lis and gradient
        _shape = [apr.org_dims(i) for i in range(2, -1, -1)]
        lis = constant_upsample(lis, _shape)
        grad = constant_upsample(grad, _shape)

        # check constant reconstruction error
        recon = pyapr.numerics.reconstruction.reconstruct_constant(apr, parts).astype(np.float32)
        img = img.astype(np.float32)
        err = np.divide(np.abs(img - recon), lis)

        # ignore regions below pipeline threshold parameters
        err[grad < self.apr_parameters.grad_th] = 0
        err[img < self.apr_parameters.Ip_th] = 0

        # maximum pixelwise error
        max_err = err.max()
        self.assertGreater(rel_err, max_err)
        return max_err

    def test_get_apr(self):
        print('Testing APR error bound')
        for ndim, path in zip([3, 2, 1], [self.impath_3D, self.impath_2D, self.impath_1D]):
            for dtype in (np.uint16, np.float32):
                typestr = 'uint16' if dtype == np.uint16 else 'float32'
                for rel_err in (0.2, 0.1, 0.05, 0.01):
                    print('{}D - {:7s} - relative error threshold E = {:1.2f} ... '.format(ndim, typestr, rel_err), end="")
                    max_err = self.__convert_image(path, dtype, rel_err=rel_err)
                    print('Maximum error = {:3.4f} -- OK!'.format(max_err))

        with self.assertRaises(TypeError):
            self.__convert_image(self.impath_1D, np.float64)

        with self.assertRaises(TypeError):
            self.__convert_image(self.impath_1D, np.uint8)

        with self.assertRaises(TypeError):
            self.__convert_image(self.impath_1D, int)

    def test_io(self):
        print('Testing APR file IO')
        self.apr_parameters.output_steps = False
        # writes files to this path, later removes the file # TODO: find a better way to do this
        fpath = os.path.join(self.data_dir, 'temporary_apr_file.apr')

        for ndim, impath in zip([1, 2, 3], [self.impath_1D, self.impath_2D, self.impath_3D]):
            print('{}D ... '.format(ndim), end="")
            img = skio.imread(impath)
            apr, parts = pyapr.converter.get_apr(img, verbose=False, params=self.apr_parameters)

            pyapr.io.write(fpath, apr, parts)

            apr2 = pyapr.APR()
            parts2 = pyapr.ShortParticles()

            pyapr.io.read(fpath, apr2, parts2)

            for i in range(3):
                self.assertEqual(apr.org_dims(i), apr2.org_dims(i))
            self.assertEqual(apr.total_number_particles(), apr2.total_number_particles())
            self.assertEqual(apr.total_number_particles(), len(parts2))

            self.assertTrue(
                np.alltrue(
                    np.array(parts, copy=False) == np.array(parts2, copy=False)
                )
            )
            print('OK!')

        os.remove(fpath)    # remove temporary file


if __name__ == '__main__':
    unittest.main()
