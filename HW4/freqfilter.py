from PIL import Image
import numpy as np

class Frequencyfilter():
    '''
    A class of functions for filtering in the frequency domain.
    '''
    
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = np.array(Image.open(img_path).convert('L'))  # read as grayscale  
        self.shape = self.img.shape  # the original size of the image, a tuple (m, n), m rows and n columns
        self.nrow, self.ncol = self.shape
    
    def show_image(self, img=None, save_path=None, modified=0):
        '''
        Show the image.
        
        Parameters:
            - img: the input image, a 2D numpy array, by default None
            - save_path: the path to save the image, a string
            - modified: whether the image will be log-transformed (1), truncated (2) or scaled (3); 0 by default.
        '''
        if img is None:
            img = self.img

        if modified==1:
            img = np.log(1+img)
            if np.max(img) != 0:
                img = 255 *(img-np.min(img))/(np.max(img)-np.min(img))
        elif modified==2:
            img = np.clip(img, 0, 255)
        elif modified==3:
            scaled = (img-np.min(img))/(np.max(img)-np.min(img))
            img = 255 *scaled
        
        new_img = Image.fromarray(img.astype(np.uint8))
        if save_path:
            new_img.save(save_path)
        else:
            new_img.show()
    
    def show_spectrum(self, img=None, save_path=None):
        '''
        Show the frequency spectrum of the image.
        
        Parameters:
            - img: the input image, a 2D numpy array, by default None
            - save_path: the path to save the spectrum, a string
        '''
        if img is None:
            img = self.img
        f = np.fft.fft2(img)
        f = np.fft.fftshift(f)
        spectrum = np.abs(f)
        self.show_image(img=spectrum, save_path=save_path, modified=1)
    
    def padding(self, img=None):
        '''
        Pad the 2-D image with zeros to avoid the wrap-around effect. The size of the padded array is 2m*2n.
        
        Parameters:
            - img: the input image (original, not padded), a 2D numpy array, by default None

        Returns:
            - img_pad: the padded array
        '''
        if img is None:
            img = self.img
        m, n = self.shape
        img_pad = np.zeros((2*m, 2*n))
        img_pad[:m, :n] = img
        return img_pad
    
    def freqfilter(self, filter_transfun):
        '''
        Apply the frequency filter to the image in the frequency domain.
        
        Parameters:
            - img: the input image (original, not padded), a 2D numpy array
            - filter_transfun: the filter transfer function, a 2D numpy array with the same size as the image
        
        Returns:
            - img_filtered: the filtered image
        '''
        m, n = self.shape
        p = 2*m
        q = 2*n  # the size of the padded array
        # step1: padding
        img_pad = self.padding()
        
        # step2: do the DFT and shift the image to the center
        for x in range(p):
            for y in range(q):
                img_pad[x, y] *= (-1)**(x+y)
        f = np.fft.fft2(img_pad)
        
        # step3: apply the filter transfer function to the DFT of the image
        g = f * filter_transfun  # element-wise multiplication
        
        # step4: do the inverse DFT and shift back
        img_filtered = np.fft.ifft2(g)
        img_filtered = np.real(img_filtered)
        for x in range(p):
            for y in range(q):
                img_filtered[x, y] *= (-1)**(x+y)

        # step5: crop the filtered image
        img_filtered = img_filtered[:m, :n]
        return img_filtered
        
    def ilpf(self, d0):
        '''
        Ideal low pass filter (ILPF).
        
        Parameters:
            - d0: the cutoff frequency
            
        Returns:
            - filter_transfun: the filter transfer function of ILPF, with size 2m*2n
        '''
        p = 2*self.nrow
        q = 2*self.ncol
        filter_transfun = np.zeros((p, q))
        for u in range(p):
            for v in range(q):
                if (u-p/2)**2 + (v-q/2)**2 <= d0**2:
                    filter_transfun[u, v] = 1
        return filter_transfun


    def glpf(self, do):
        '''
        Gaussian low pass filter (GLPF).
        
        Parameters:
            - d0: the cutoff frequency, also equals to the standard deviation of the Gaussian function 
            
        Returns:
            - filter_transfun: the filter transfer function of GLPF, with size 2m*2n
        '''
        p = 2*self.nrow
        q = 2*self.ncol
        filter_transfun = np.zeros((p, q))
        for u in range(p):
            for v in range(q):
                filter_transfun[u, v] = np.exp(-((u-p/2)**2 + (v-q/2)**2)/(2*do**2))
        return filter_transfun
    
    def ihpf(self, d0):
        '''
        Ideal high pass filter (IHPF).
        
        Parameters:
            - d0: the cutoff frequency
            
        Returns:
            - filter_transfun: the filter transfer function of IHPF, with size 2m*2n
        '''
        return 1 - self.ilpf(d0)
    
    def ghpf(self, d0):
        '''
        Gaussian high pass filter (GHPF).
        
        Parameters:
            - d0: the cutoff frequency, also equals to the standard deviation of the Gaussian function 
            
        Returns:
            - filter_transfun: the filter transfer function of GHPF, with size 2m*2n
        '''
        return 1 - self.glpf(d0)
    
    def laplacian_filter(self):
        '''
        Laplacian filter for image sharpening in the frequency domain.
            
        Returns:
            - filter_transfun: the filter transfer function of Laplacian, with size p*q
        '''
        p = 2*self.nrow
        q = 2*self.ncol
        filter_transfun = np.zeros((p, q))
        for u in range(p):
            for v in range(q):
                    filter_transfun[u, v] = -4 * np.pi**2 * ((u-p/2)**2 + (v-q/2)**2)
        return filter_transfun
    
    def laplacian_sharpen(self):
        '''
        Laplacian sharpening in the frequency domain.
        
        Returns:
            - img_sharpened: the sharpened image
        '''
        img = self.img
        m, n = self.shape
        # scale the image to [0, 1]
        img_scaled = img / 255
        img_pad = self.padding(img_scaled)
        f = np.fft.fft2(img_pad)
        f = np.fft.fftshift(f)
        h = self.laplacian_filter()
        # the second derivative of f
        f2 = f * h
        f2 = np.fft.ifftshift(f2)
        f2 = np.fft.ifft2(f2)
        f2 = np.real(f2)
        # scale the second derivative to [-1, 1]
        oldrange = np.max(f2) - np.min(f2)
        newrange = 2
        f2scaled = (f2 - np.min(f2)) * newrange / oldrange - 1
        img_sharpened = img_pad - f2scaled
        img_sharpened = np.clip(img_sharpened, 0, 1)
        return img_sharpened[:m, :n]


# # smoothing
# img_path = './test_pattern_blurring.tif'
# img = Frequencyfilter(img_path)
# img.show_spectrum()
# # new = img.freqfilter(img.ilpf(60))
# new = img.freqfilter(img.glpf(30))
# # img.show_spectrum(img=new)
# img.show_spectrum(img=new)
# img.show_image(img=new, modified=3)



# # sharpening
# # using ihpf or ghpf
# img_path = './blown_ic.tif'
# img = Frequencyfilter(img_path)
# img.show_spectrum()
# # new = img.freqfilter(img.ihpf(30))
# new = img.freqfilter(img.ghpf(30))
# out = 3*new + img.img
# img.show_image(img=out, modified=2)

# # using Laplacian filter
# img_path = './blurry_moon.tif'
# img = Frequencyfilter(img_path)
# img.show_spectrum()
# new = img.laplacian_sharpen()
# img.show_image(img=new, modified=3)