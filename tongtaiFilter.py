import numpy as np
from PIL import Image


def test(height, width, imgg):
    # 对图像取自然对数
    rows, cols = height, width
    img_log = np.log(imgg + 1)
    img_fft = np.fft.fft2(img_log)
    img_fftshift = np.fft.fftshift(img_fft)
    dst_fftshift = np.zeros_like(img_fftshift)
    d0 = 10
    r1 = 0.5
    rh = 2
    c = 4
    h = 2.0
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * img_fftshift
    dst_fftshift = (h - 1) * dst_fftshift + 1
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst1 = np.exp(dst)
    dst = np.uint8(np.clip(dst1, 0, 255))
    finalImage = Image.fromarray(dst).convert('L')
    finalImage16 = dst1
    return finalImage, finalImage16
