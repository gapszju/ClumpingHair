import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# set OPENCV_IO_ENABLE_OPENEXR environ value before cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


class LogGabor():
    """
    Defines a LogGabor framework by defining a ``loggabor`` function which return the envelope of a log-Gabor filter.

    Its envelope is equivalent to a log-normal probability distribution on the frequency axis, and von-mises on the radial axis.

    """
    def __init__(self, ksize, sf_0, B_sf, B_theta):
        """
        Parameters
        ----------

        ksize : tuple of ints
            size of the kernel (in pixels)

        sf_0 : float
            peak spatial frequency (cycles per pixel)

        B_sf : float
            spatial frequency bandwidth (octaves)

        B_theta : float
            orientation bandwidth (radians)
        
        """
        self.ksize = ksize
        self.sf_0 = sf_0
        self.B_sf = B_sf
        self.B_theta = B_theta
        
        # precompute the grid of frequencies and orientations
        self.f, self.f_theta = self.fourier_grid()
        self.f[self.ksize[0]//2, self.ksize[1]//2] = 1e-12 # to avoid errors when dividing by frequency
        self.f_mask = self.retina()
        

    def fourier_grid(self):
        """
            use that function to define a reference frame for envelopes in Fourier space.
            In general, it is more efficient to define dimensions as powers of 2.

        """
        # From the numpy doc:
        # (see http://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft )
        # The values in the result follow so-called “standard” order: If A = fft(a, n),
        # then A[0] contains the zero-frequency term (the mean of the signal), which
        # is always purely real for real inputs. Then A[1:n/2] contains the positive-frequency
        # terms, and A[n/2+1:] contains the negative-frequency terms, in order of
        # decreasingly negative frequency. For an even number of input points, A[n/2]
        # represents both positive and negative Nyquist frequency, and is also purely
        # real for real input. For an odd number of input points, A[(n-1)/2] contains
        # the largest positive frequency, while A[(n+1)/2] contains the largest negative
        # frequency. The routine np.fft.fftfreq(A) returns an array giving the frequencies
        # of corresponding elements in the output. The routine np.fft.fftshift(A) shifts
        # transforms and their frequencies to put the zero-frequency components in the
        # middle, and np.fft.ifftshift(A) undoes that shift.
        #

        fx = fftshift(np.fft.fftfreq(self.ksize[1]))
        fy = fftshift(np.fft.fftfreq(self.ksize[0]))

        X, Y = np.meshgrid(fx, fy, indexing='xy')
        R = np.sqrt(X**2 + Y**2)
        Theta = -np.arctan2(Y, X)
        
        return R, Theta
    
    def retina(self, df=.07, sigma=.5):
        """
        A parametric description of the envelope of retinal processsing.
        See http://blog.invibe.net/posts/2015-05-21-a-simple-pre-processing-filter-for-image-processing.html
        for more information.

        In digital images, some of the energy in Fourier space is concentrated outside the
        disk corresponding to the Nyquist frequency. Let's design a filter with:

            - a sharp cut-off for radial frequencies higher than the Nyquist frequency,
            - times a smooth but sharp transition (implemented with a decaying exponential),
            - times a high-pass filter designed by one minus a gaussian blur.

        This filter is rotation invariant.

        Note that this filter is defined by two parameters:
            - one for scaling the smoothness of the transition in the high-frequency range,
            - one for the characteristic length of the high-pass filter.

        The first is defined relative to the Nyquist frequency (in absolute values) while the second
        is relative to the size of the image in pixels and is given in number of pixels.
        """
        # removing high frequencies in the corners
        env = (1-np.exp((self.f-.5)/(.5*df)))*(self.f<.5)
        # removing low frequencies
        env *= 1-np.exp(-.5*(self.f**2)/((sigma/self.ksize[1])**2))
        return env
    
    def invert(self, FT_image, full=False):
        if full:
            return ifft2(ifftshift(FT_image))
        else:
            return ifft2(ifftshift(FT_image)).real

    ## LOW LEVEL OPERATIONS
    def band(self):
        """
        Returns the radial frequency envelope:

        Selects a preferred spatial frequency ``sf_0`` and a bandwidth ``B_sf``.

        """
        if self.sf_0 == 0.:
            return 1.
        else:
            # see http://en.wikipedia.org/wiki/Log-normal_distribution
            env = 1./self.f*np.exp(-.5*(np.log(self.f/self.sf_0)**2)/self.B_sf**2)
        return env

    def orientation(self, theta):
        """
        Returns the orientation envelope:
        We use a von-Mises distribution on the orientation:
        - mean orientation is ``theta`` (in radians),
        - ``B_theta`` is the bandwidth (in radians). It is equal to the standard deviation of the Gaussian
        envelope which approximate the distribution for low bandwidths. The Half-Width at Half Height is
        given by approximately np.sqrt(2*B_theta_**2*np.log(2)).

        # selecting one direction,  theta is the mean direction, B_theta the spread
        # we use a von-mises distribution on the orientation
        # see http://en.wikipedia.org/wiki/Von_Mises_distribution
        """
        if self.B_theta is np.inf: # for large bandwidth, returns a strictly flat envelope
            enveloppe_orientation = 1.
        else:
            # As shown in:
            #  http://www.csse.uwa.edu.au/~pk/research/matlabfns/PhaseCongruency/Docs/convexpl.html
            # this single bump allows (without the symmetric) to code both symmetric
            # and anti-symmetric parts in one shot.
            cos_angle = np.cos(self.f_theta-theta)
            enveloppe_orientation = np.exp(cos_angle/self.B_theta**2)
        return enveloppe_orientation

    ## MID LEVEL OPERATIONS
    def loggabor(self, theta, preprocess=True):
        """
        Returns the envelope of a LogGabor

        Note that the convention for coordinates follows that of matrices:
        the origin is at the top left of the image, and coordinates are first
        the rows (vertical axis, going down) then the columns (horizontal axis,
        going right).

        """
        env = self.band() * self.orientation(theta)
        
        if preprocess: # retina processing
            env *= self.f_mask 
            
        # normalizing energy:
        env /= np.sqrt((np.abs(env)**2).mean())
        # in the case a a single bump (see ``orientation``), we should compensate the fact that the distribution gets complex:
        env *= np.sqrt(2.)
        
        return env

    def loggabor_image(self, theta, phase=0.):
        """
        Returns the image of a LogGabor

        Note that the convention for coordinates follows that of matrices:
        the origin is at the top left of the image, and coordinates are first
        the rows (vertical axis, going down) then the columns (horizontal axis,
        going right).

        """
        FT_lg = self.loggabor(theta=theta)
        FT_lg = FT_lg * np.exp(1j * phase)
        return fftshift(self.invert(FT_lg, full=False))

    def show_loggabor(self, theta, phase=0.):
        FT_image = self.loggabor(theta)
        image = fftshift(self.invert(FT_image))
        
        fig = plt.figure(figsize=(14*self.ksize[1]/self.ksize[0], 14/2))
        a1 = fig.add_subplot(121)
        a2 = fig.add_subplot(122)
        
        a1.imshow(np.absolute(FT_image)/np.absolute(FT_image).max()*2-1, cmap="hot")
        a2.imshow(image, cmap="gray")
        plt.setp(a1, title='Spectrum')
        plt.setp(a2, title='Image')
        plt.setp(a1, yticks=[self.ksize[0]/2], xticks=[self.ksize[1]/2], xticklabels=[''], yticklabels=[''])
        plt.setp(a2, xticks=[], yticks=[])
        a1.axis('equal')
        a2.axis('equal')
        
        return fig, a1, a2
    
    def show_image_response(self, image, theta):
        assert image.shape == self.ksize, "image and kernel must have the same size"
        
        image_ft = fftshift(fft2(image))
        FT_image = image_ft * self.loggabor(theta)
        image_filtered = self.invert(FT_image, full=True)
        
        fig = plt.figure()
        a1 = fig.add_subplot(121)
        a2 = fig.add_subplot(122)
        
        ca1 = a1.imshow(image, cmap="gray")
        ca2 = a2.imshow(np.abs(image_filtered), cmap="hot")
        plt.setp(a1, title='Source Image')
        plt.setp(a2, title='Filtered Amplitude')
        plt.setp(a1, xticks=[], yticks=[])
        plt.setp(a2, xticks=[], yticks=[])
        plt.colorbar(ca2)
        
        return fig, a1, a2
    
    def show_pixel_response_curve(self, image, pos_x, pos_y):
        assert image.shape == self.ksize, "image and kernel must have the same size"
        
        image_ft = fftshift(fft2(image))

        x = np.linspace(0, np.pi, 32, endpoint=False)
        y1 = []
        y2 = []
        for theta in x:
            print("generating filter for angle: ", theta)
            kernel_ft = lg.loggabor(theta=theta)
            filtered_ft = image_ft * kernel_ft
            filtered = lg.invert(filtered_ft, full=True)
            y1 += np.abs(filtered[pos_y, pos_x]),
            y2 += np.abs(np.real(filtered[pos_y, pos_x])),

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(image[pos_y-10:pos_y+10, pos_x-10:pos_x+10])
        axes[0].set_title("Source Image (zoom)")
        axes[1].plot(x*180/np.pi, y1), axes[1].set_title("Amplitude")
        axes[2].plot(x*180/np.pi, y2), axes[2].set_title("Real part")
        
        return fig, *axes
    

def invert_image(img):
    if img.dtype == np.uint8:
        return 255 - img
    elif img.dtype == np.float32 or img.dtype == np.float64:
        return 1.0 - img
    else:
        raise ValueError("Unsupported image data type.")


def calc_orientation_map(image, bin_size=32, sf_0=1/3, B_sf=np.log(2), B_theta=0.1):
    """Returns the orientation map and confidence map of the given image.
    """
    image = invert_image(image)
    image_ft = fftshift(fft2(image))
    lg = LogGabor(ksize=image.shape, sf_0=sf_0, B_sf=B_sf, B_theta=B_theta)
    
    responses_abs = []
    responses_real = []
    thetas = np.linspace(0, np.pi, bin_size, endpoint=False)
    for theta in thetas:
        print("generating filter for angle: ", theta)
        kernel_ft = lg.loggabor(theta=theta)
        filtered_ft = image_ft * kernel_ft
        filtered = lg.invert(filtered_ft, full=True)
        responses_abs += np.abs(filtered),
        responses_real += np.real(filtered),
        
    orientation_map = np.array(responses_abs).argmax(axis=0) / bin_size * np.pi
    confidence_map = np.array(responses_real).max(axis=0)
    
    # indices = thetas[:, np.newaxis, np.newaxis]
    # mean = np.sum(indices * responses, axis=0) / np.sum(responses, axis=0)
    # var2 = np.sum(((indices - mean) ** 2) * responses, axis=0) / np.sum(responses, axis=0)
    # confidence_map = 1 / var2
    
    return orientation_map, confidence_map


def orientation_map_gabor(image, bin_size=32, sigma=1.2, gamma=0.75, lambd=3):
    image = invert_image(image)
    thetas = np.linspace(0, np.pi, bin_size, endpoint=False)
    responses = []
    for theta in thetas:
        print("generating filter for angle: ", theta)
        kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        responses += cv2.filter2D(image, cv2.CV_32F, kernel),
    responses = np.array(responses)
    max_response = np.max(responses, axis=0)
    orientation_map = np.argmax(responses, axis=0) / bin_size * np.pi
    return orientation_map, max_response


# tests
if __name__ == "__main__":
    image_path = "X:/hairstep/Real_image2/resized_img/tim-mossholder-_V4qRy2dphk-unsplash.png"
    strand_map_path = "X:/hairstep/Real_image2/strand_map/tim-mossholder-_V4qRy2dphk-unsplash.png"
    output_path =  "X:/hairstep/Real_image2/strand_map_gabor/tim-mossholder-_V4qRy2dphk-unsplash.png"
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0
    strand_map = cv2.imread(strand_map_path) / 255.0
    mask = strand_map[..., 2] > 0.99
    ref_orien = strand_map[..., :2] *2 - 1
    
    ori_map, conf_map = calc_orientation_map(image, bin_size=128, sf_0=1/0.5)
    strand_map_gabor = np.stack([np.sin(ori_map), -np.cos(ori_map)], axis=-1)
    
    sign = np.sign((strand_map_gabor * ref_orien).sum(axis=-1))
    strand_map_gabor = (strand_map_gabor * sign[..., np.newaxis] * 0.5 + 0.5) * mask[..., np.newaxis]
    strand_map_gabor = np.concatenate([strand_map_gabor, strand_map[..., 2:]], axis=-1)
    
    plt.imshow(strand_map_gabor.clip(0, 1)[..., ::-1])
    plt.show()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, strand_map_gabor.clip(0, 1)[..., ::-1])