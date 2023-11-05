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
    
    def show_spectrum(self, save_path=None):
        '''
        Show the frequency spectrum of the image.
        
        Parameters:
            - save_path: the path to save the spectrum, a string
        '''
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
    
    def filternopad(self, filter_transfun):
        '''
        Apply the frequency filter to the image in the frequency domain without padding.
        
        Parameters:
            - img: the input image (original, not padded), a 2D numpy array
            - filter_transfun: the filter transfer function, a 2D numpy array with the same size as the image
        
        Returns:
            - img_filtered: the filtered image
        '''
        img = self.img
        
        # step2: do the DFT and shift the image to the center
        f = np.fft.fft2(img)
        f = np.fft.fftshift(f)
        
        # step3: apply the filter transfer function to the DFT of the image
        g = f * filter_transfun  # element-wise multiplication
        
        # step4: do the inverse DFT and shift back
        img_filtered = np.fft.ifftshift(g)
        img_filtered = np.fft.ifft2(img_filtered)
        img_filtered = np.real(img_filtered)

        return img_filtered
    
    def ibrf(self, c0, w):
        '''
        Ideal band reject filter (IBRF).
        
        Parameters:
            - c0: center of the band
            - w: width of the frequency band
            
        Returns:
            - filter_transfun: the filter transfer function of IBRF, with size 2m*2n
        '''
        p = 2*self.nrow
        q = 2*self.ncol
        filter_transfun = np.ones((p, q))
        for u in range(p):
            for v in range(q):
                d = np.sqrt((u-p/2)**2 + (v-q/2)**2)
                if c0-w/2 <= d <= c0+w/2:
                    filter_transfun[u, v] = 0
        return filter_transfun
    
    def gbrf(self, c0, w):
        '''
        Gaussian band reject filter (GBRF).
        
        Parameters:
            - c0: center of the band
            - w: width of the frequency band
            
        Returns:
            - filter_transfun: the filter transfer function of GBRF, with size 2m*2n
        '''
        p = 2*self.nrow
        q = 2*self.ncol
        filter_transfun = np.ones((p, q))
        for u in range(p):
            for v in range(q):
                d2 = (u-p/2)**2 + (v-q/2)**2
                filter_transfun[u, v] = 1 - np.exp(-(d2 - c0**2)**2 / (d2*w**2))
        return filter_transfun
    
    def ibpf(self, c0, w):
        '''
        Ideal band pass filter (IBPF).
        
        Parameters:
            - c0: center of the band
            - w: width of the frequency band
            
        Returns:
            - filter_transfun: the filter transfer function of IBPF, with size 2m*2n
        '''
        return 1 - self.ibrf(c0, w)
    
    def gbpf(self, c0, w):
        '''
        Gaussian band pass filter (GBPF).
        
        Parameters:
            - c0: center of the band
            - w: width of the frequency band
            
        Returns:
            - filter_transfun: the filter transfer function of GBPF, with size 2m*2n
        '''
        return 1 - self.gbrf(c0, w)
    
    def ghpf_shift(self, d0, u0, v0):
        '''
        Gaussian high pass filter (GHPF) with center shifted to (u0, v0).
        
        Parameters:
            - d0: the cutoff frequency
            - u0, v0: the center coordinates of the highpass filter
            
        Returns:
            - filter_transfun: the filter transfer function of GHPF, with size 2m*2n
        '''
        p = self.nrow
        q = self.ncol
        filter_transfun = np.zeros((p, q))
        for u in range(p):
            for v in range(q):
                d2 = (u-u0)**2 + (v-v0)**2
                filter_transfun[u, v] = 1 - np.exp(-d2/(2*d0**2))
        return filter_transfun
    
    def bhpf_shift(self, d0, u0, v0, n):
        '''
        Butterworth high pass filter (BHPF) with center shifted to (u0, v0).
        
        Parameters:
            - d0: the cutoff frequency
            - u0, v0: the center coordinates of the highpass filter
            - n: the order of the filter
            
        Returns:
            - filter_transfun: the filter transfer function of BHPF, with size 2m*2n
        '''
        p = self.nrow
        q = self.ncol
        filter_transfun = np.zeros((p, q))
        for u in range(p):
            for v in range(q):
                d2 = (u-u0)**2 + (v-v0)**2
                filter_transfun[u, v] = 1 / (1 + (d0**2/d2)**n)
        return filter_transfun
    
    def notch_reject(self, coord, d0, n=1):
        '''
        Notch reject filter.
        
        Parameters:
            - coord: the center coordinates of each highpass filter, k*2 array, k is the number of filters
            - d0: the cutoff frequency of the highpass filter
            
        Returns:
            - filter_transfun: the filter transfer function of notch reject filter, with size 2m*2n
        '''
        m = self.nrow
        n = self.ncol
        k = coord.shape[0]
        nr = np.ones((m,n))
        for i in range(k):
            u, v = coord[i]
            # nr *= self.ghpf_shift(d0, u, v) * self.ghpf_shift(d0, -u, -v)
            nr *= self.bhpf_shift(d0, u, v, n) * self.bhpf_shift(d0, -u, -v, n)
        return nr
    
    def notch_pass(self, coord, d0, n=1):
        '''
        Notch pass filter.
        
        Parameters:
            - coord: the center coordinates of each highpass filter, k*2 array, k is the number of filters
            - d0: the cutoff frequency of the highpass filter
            
        Returns:
            - filter_transfun: the filter transfer function of notch pass filter, with size 2m*2n
        '''
        return 1 - self.notch_reject(coord, d0, n)

# img_path = './HW4/test_pattern_blurring.tif'
# img_path = './HW4/blurry_moon.tif'
# img_path = './HW4/blown_ic.tif'
# img_path = './HW4/car.tif'
# img_path = './HW4/cassini.tif'
# img = Frequencyfilter(img_path)
# img.show_spectrum()


# new = img.filternopad(img.notch_reject(np.array([[82, 56],[164, 112]]), 3, 4))
# img.show_image(img=new, modified=2)