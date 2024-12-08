import cv2 as cv
from typing import Union, cast

class Blender:
    """https://docs.opencv.org/4.x/d6/d4a/classcv_1_1detail_1_1Blender.html"""

    BLENDER_CHOICES = (
        "multiband",
        "feather",
        "no",
    )
    DEFAULT_BLENDER = "multiband"
    DEFAULT_BLEND_STRENGTH = 5

    def __init__(self, blender_type=DEFAULT_BLENDER, blend_strength=DEFAULT_BLEND_STRENGTH):
        self.blender_type = blender_type
        self.blend_strength = blend_strength
        self.blender = self.create_blender(blender_type, blend_strength)

    def create_blender(self, blender_type: str, blend_strength: int) -> Union[cv.detail.Blender, cv.detail.FeatherBlender, cv.detail.MultiBandBlender]:
        if blender_type == "no":
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
        elif blender_type == "feather":
            blender = cast(cv.detail.FeatherBlender, cv.detail.Blender_createDefault(cv.detail.Blender_FEATHER))
            # blender.setSharpness(blend_strength / 100.0)
        elif blender_type == "multiband":
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_MULTI_BAND)
        else:
            raise ValueError(f"Unknown blender type {blender_type}")
        
        return blender

    def prepare(self, corners, sizes):
        sizes = [(int(s[0]), int(s[1])) for s in sizes]  # Ensure sizes are integers
        self.blender.prepare(corners, sizes)

    def feed(self, img, mask, corner):
        if img is None:
            raise ValueError("Image passed to feed is None")
        if mask is None:
            raise ValueError("Mask passed to feed is None")
        if corner is None:
            raise ValueError("Corner passed to feed is None")
        
        # Convert img to 16SC3 if it is not already
        if img.dtype != 'int16' or img.shape[2] != 3:
            img = img.astype('int16')

        # Ensure the mask is of type CV_8UC1
        if mask.dtype != 'uint8' or len(mask.shape) != 2:
            raise ValueError("Mask must be of type CV_8UC1")

        self.blender.feed(cv.UMat(img), cv.UMat(mask), corner)

    def blend(self):
        result, result_mask = self.blender.blend(None, None)
        return result, result_mask

    @classmethod
    def create_panorama(cls, imgs, masks, corners, sizes):
        # Use feather blender when blending multiple images
        blender = cls("feather")
        blender.prepare(corners, sizes)
        for img, mask, corner in zip(imgs, masks, corners):
            blender.feed(img, mask, corner)
        return blender.blend()