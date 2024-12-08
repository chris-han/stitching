from statistics import median
import cv2 as cv
import numpy as np

class Warper:
    """https://docs.opencv.org/4.x/da/db8/classcv_1_1detail_1_1RotationWarper.html"""

    WARP_TYPE_CHOICES = (
        "spherical",
        "plane",
        "affine",
        "cylindrical",
        "fisheye",
        "stereographic",
        "compressedPlaneA2B1",
        "compressedPlaneA1.5B1",
        "compressedPlanePortraitA2B1",
        "compressedPlanePortraitA1.5B1",
        "paniniA2B1",
        "paniniA1.5B1",
        "paniniPortraitA2B1",
        "paniniPortraitA1.5B1",
        "mercator",
        "transverseMercator",
    )

    DEFAULT_WARP_TYPE = "spherical"

    def __init__(self, warper_type=DEFAULT_WARP_TYPE):
        self.warper_type = warper_type
        self.scale = None

    def set_scale(self, cameras):
        focals = [cam.focal for cam in cameras]
        self.scale = median(focals)

    def warp_images(self, imgs, cameras, aspect=1):
        for img, camera in zip(imgs, cameras):
            yield self.warp_image(img, camera, aspect)

    def warp_image(self, img, camera, aspect=1):
        img = self.check_and_resize_image(img)
        h, w = img.shape[:2]
        print(f"Warping image with dimensions {w}x{h}")
        warper = cv.PyRotationWarper(self.warper_type, self.scale * aspect)
        try:
            _, warped_image = warper.warp(
                img,
                Warper.get_K(camera, aspect),
                camera.R,
                cv.INTER_LINEAR,
                cv.BORDER_REFLECT,
            )
            print(f"Warped image shape: {warped_image.shape}, type: {warped_image.dtype}")
        except cv.error as e:
            print(f"Error during warping: {e}")
            raise
        return warped_image

    def create_and_warp_masks(self, sizes, cameras, aspect=1):
        warped_masks = []
        for size, camera in zip(sizes, cameras):
            print(f"Creating and warping mask for size: {size}, camera: {camera}")
            try:
                warped_mask = self.create_and_warp_mask(size, camera, aspect)
                warped_masks.append(warped_mask)
                print(f"Warped mask shape: {warped_mask.shape}, type: {warped_mask.dtype}")
            except cv.error as e:
                print(f"Error during mask warping: {e}")
                raise
        return warped_masks

    def create_and_warp_mask(self, size, camera, aspect=1):
        warper = cv.PyRotationWarper(self.warper_type, self.scale * aspect)
        mask = 255 * np.ones((size[1], size[0]), np.uint8)
        mask = self.check_and_resize_image(mask)
        h, w = mask.shape[:2]
        print(f"Warping mask with dimensions {w}x{h}")
        try:
            _, warped_mask = warper.warp(
                mask,
                Warper.get_K(camera, aspect),
                camera.R,
                cv.INTER_NEAREST,
                cv.BORDER_CONSTANT,
            )
        except cv.error as e:
            print(f"Error during warping mask: {e}")
            raise
        return warped_mask

    def warp_rois(self, sizes, cameras, aspect=1):
        roi_corners = []
        roi_sizes = []
        for size, camera in zip(sizes, cameras):
            roi = self.warp_roi(size, camera, aspect)
            roi_corners.append(roi[0:2])
            roi_sizes.append(roi[2:4])
        return roi_corners, roi_sizes

    def warp_roi(self, size, camera, aspect=1):
        warper = cv.PyRotationWarper(self.warper_type, self.scale * aspect)
        K = Warper.get_K(camera, aspect)
        return warper.warpRoi(size, K, camera.R)

    @staticmethod
    def get_K(camera, aspect=1):
        K = camera.K().astype(np.float32)
        K[0, 0] *= aspect
        K[0, 2] *= aspect
        K[1, 1] *= aspect
        K[1, 2] *= aspect
        print(f"K matrix for camera: {K}")
        return K

    @staticmethod
    def check_and_resize_image(image):
        max_dimension = 32766  # Just below SHRT_MAX
        h, w = image.shape[:2]
        if h >= max_dimension or w >= max_dimension:
            scaling_factor = min(max_dimension / h, max_dimension / w)
            new_w, new_h = int(w * scaling_factor), int(h * scaling_factor)
            print(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
            image = cv.resize(image, (new_w, new_h))
        return image