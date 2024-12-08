import warnings
from types import SimpleNamespace

from .blender import Blender
from .camera_adjuster import CameraAdjuster
from .camera_estimator import CameraEstimator
from .camera_wave_corrector import WaveCorrector
from .cropper import Cropper
from .exposure_error_compensator import ExposureErrorCompensator
from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .images import Images
from .seam_finder import SeamFinder
from .stitching_error import StitchingError, StitchingWarning
from .subsetter import Subsetter
from .timelapser import Timelapser
from .verbose import verbose_stitching
from .warper import Warper
import numpy as np
import cv2
class Stitcher:
    DEFAULT_SETTINGS = {
        "medium_megapix": Images.Resolution.MEDIUM.value,
        "detector": FeatureDetector.DEFAULT_DETECTOR,
        "nfeatures": 500,
        "matcher_type": FeatureMatcher.DEFAULT_MATCHER,
        "range_width": FeatureMatcher.DEFAULT_RANGE_WIDTH,
        "try_use_gpu": False,
        "match_conf": None,
        "confidence_threshold": Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
        "matches_graph_dot_file": Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
        "estimator": CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
        "adjuster": CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
        "refinement_mask": CameraAdjuster.DEFAULT_REFINEMENT_MASK,
        "wave_correct_kind": WaveCorrector.DEFAULT_WAVE_CORRECTION,
        "warper_type": Warper.DEFAULT_WARP_TYPE,
        "low_megapix": Images.Resolution.LOW.value,
        "crop": Cropper.DEFAULT_CROP,
        "compensator": ExposureErrorCompensator.DEFAULT_COMPENSATOR,
        "nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
        "block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
        "finder": SeamFinder.DEFAULT_SEAM_FINDER,
        "final_megapix": Images.Resolution.FINAL.value,
        "blender_type": Blender.DEFAULT_BLENDER,
        "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
        "timelapse": Timelapser.DEFAULT_TIMELAPSE,
        "timelapse_prefix": Timelapser.DEFAULT_TIMELAPSE_PREFIX,
    }
    
    def __init__(self, **kwargs):
        self.initialize_stitcher(**kwargs)

    def initialize_stitcher(self, **kwargs):
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.validate_kwargs(kwargs)
        self.kwargs = kwargs
        self.settings.update(kwargs)

        args = SimpleNamespace(**self.settings)
        # Initialize instance variables with settings
        self.medium_megapix = args.medium_megapix
        self.low_megapix = args.low_megapix
        self.final_megapix = args.final_megapix

        if args.detector in ("orb", "sift"):
            self.detector = FeatureDetector(args.detector, nfeatures=args.nfeatures)
        else:
            self.detector = FeatureDetector(args.detector)

        match_conf = FeatureMatcher.get_match_conf(args.match_conf, args.detector)
        self.matcher = FeatureMatcher(
            args.matcher_type,
            args.range_width,
            try_use_gpu=args.try_use_gpu,
            match_conf=match_conf,
        )

        self.subsetter = Subsetter(
            args.confidence_threshold, args.matches_graph_dot_file
        )

        self.camera_estimator = CameraEstimator(args.estimator)
        self.camera_adjuster = CameraAdjuster(
            args.adjuster, args.refinement_mask, args.confidence_threshold
        )
        self.wave_corrector = WaveCorrector(args.wave_correct_kind)
        self.warper = Warper(args.warper_type)
        self.cropper = Cropper(args.crop)
        self.compensator = ExposureErrorCompensator(
            args.compensator, args.nr_feeds, args.block_size
        )
        self.seam_finder = SeamFinder(args.finder)
        self.blender = Blender(args.blender_type, args.blend_strength)
        self.timelapser = Timelapser(args.timelapse, args.timelapse_prefix)

    def initialize_composition(self, corners, sizes, **kwargs):
        self.dest_roi = self.calculate_dest_roi(corners, sizes)
        if self.timelapser.do_timelapse:
            self.timelapser.initialize(corners, sizes, **kwargs)
        else:
            self.blender.prepare(corners, sizes, **kwargs)

    def calculate_dest_roi(self, corners, sizes):
        # Assuming a simple bounding box that covers all the image corners and sizes
        top_left = [min(corner[0] for corner in corners), min(corner[1] for corner in corners)]
        bottom_right = [max(corner[0] + size[0] for corner, size in zip(corners, sizes)),
                        max(corner[1] + size[1] for corner, size in zip(corners, sizes))]
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        return top_left, width, height

    def stitch(self, images, feature_masks=[], **kwargs):
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )

        imgs = self.resize_medium_resolution()
        features = self.find_features(imgs, feature_masks)
        matches = self.match_features(features)
        imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.refine_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        self.estimate_scale(cameras)

        imgs = self.resize_low_resolution(imgs)
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, cameras)
        self.prepare_cropper(imgs, masks, corners, sizes)
        imgs, masks, corners, sizes = self.crop_low_resolution(
            imgs, masks, corners, sizes
        )
        self.estimate_exposure_errors(corners, imgs, masks)
        seam_masks = self.find_seam_masks(imgs, corners, masks)

        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, cameras)
        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs, masks, corners, sizes
        )
        self.set_masks(masks)
        imgs = list(self.compensate_exposure_errors(corners, imgs))
        seam_masks = list(self.resize_seam_masks(seam_masks))

        self.initialize_composition(corners, sizes, **kwargs)
        self.blend_images(imgs, seam_masks, corners, sizes)
        return self.create_final_panorama()

    def stitch_verbose(self, images, feature_masks=[], verbose_dir=None):
        return verbose_stitching(self, images, feature_masks, verbose_dir)
    
    def resize_medium_resolution(self):
        return list(self.images.resize(Images.Resolution.MEDIUM))

    def find_features(self, imgs, feature_masks=[]):
        if len(feature_masks) == 0:
            return self.detector.detect(imgs)
        else:
            feature_masks = Images.of(
                feature_masks, self.medium_megapix, self.low_megapix, self.final_megapix
            )
            feature_masks = list(feature_masks.resize(Images.Resolution.MEDIUM))
            feature_masks = [Images.to_binary(mask) for mask in feature_masks]
            return self.detector.detect_with_masks(imgs, feature_masks)

    def match_features(self, features):
        return self.matcher.match_features(features)

    def subset(self, imgs, features, matches):
        indices = self.subsetter.subset(self.images.names, features, matches)
        imgs = Subsetter.subset_list(imgs, indices)
        features = Subsetter.subset_list(features, indices)
        matches = Subsetter.subset_matches(matches, indices)
        self.images.subset(indices)
        return imgs, features, matches

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def refine_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)

    def estimate_scale(self, cameras):
        self.warper.set_scale(cameras)

    def resize_low_resolution(self, imgs=None):
        return list(self.images.resize(Images.Resolution.LOW, imgs))

    def warp_low_resolution(self, imgs, cameras):
        sizes = self.images.get_scaled_img_sizes(Images.Resolution.LOW)
        camera_aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.LOW
        )
        imgs, masks, corners, sizes = self.warp(imgs, cameras, sizes, camera_aspect)
        return list(imgs), list(masks), corners, sizes

    def warp_final_resolution(self, imgs, cameras):
        sizes = self.images.get_scaled_img_sizes(Images.Resolution.FINAL)
        camera_aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.FINAL
        )
        return self.warp(imgs, cameras, sizes, camera_aspect)

    def warp(self, imgs, cameras, sizes, aspect=1):
        imgs = self.warper.warp_images(imgs, cameras, aspect)
        masks = self.warper.create_and_warp_masks(sizes, cameras, aspect)
        corners, sizes = self.warper.warp_rois(sizes, cameras, aspect)
        return imgs, masks, corners, sizes

    def prepare_cropper(self, imgs, masks, corners, sizes):
        self.cropper.prepare(imgs, masks, corners, sizes)

    def crop_low_resolution(self, imgs, masks, corners, sizes):
        imgs, masks, corners, sizes = self.crop(imgs, masks, corners, sizes)
        return list(imgs), list(masks), corners, sizes

    def crop_final_resolution(self, imgs, masks, corners, sizes):
        lir_aspect = self.images.get_ratio(
            Images.Resolution.LOW, Images.Resolution.FINAL
        )
        return self.crop(imgs, masks, corners, sizes, lir_aspect)

    def crop(self, imgs, masks, corners, sizes, aspect=1):
        masks = self.cropper.crop_images(masks, aspect)
        imgs = self.cropper.crop_images(imgs, aspect)
        corners, sizes = self.cropper.crop_rois(corners, sizes, aspect)
        return imgs, masks, corners, sizes

    def estimate_exposure_errors(self, corners, imgs, masks):
        self.compensator.feed(corners, imgs, masks)

    def find_seam_masks(self, imgs, corners, masks):
        return self.seam_finder.find(imgs, corners, masks)

    def resize_final_resolution(self):
        return list(self.images.resize(Images.Resolution.FINAL))

    def resize_seam_masks(self, seam_masks):
        print("[DEBUG] Resizing seam masks")
        resized_masks = []
        for idx, seam_mask in enumerate(seam_masks):
            print(f'[DEBUG] Resizing seam mask {idx}')
            mask = self.get_mask(idx)
            print(f'[DEBUG] Got mask for seam mask {idx}')
            resized_masks.append(SeamFinder.resize(seam_mask, mask))
        return resized_masks

    def compensate_exposure_errors(self, corners, imgs):
        for idx, (corner, img) in enumerate(zip(corners, imgs)):
            mask = self.get_mask(idx)
            yield self.compensator.apply(idx, corner, img, mask)

    def set_masks(self, mask_generator):
        print("[DEBUG] Setting mask generator")
        self.masks = list(mask_generator)  # Convert generator to list
        self.mask_index = -1
        print("[DEBUG] Mask index initialized to -1")

    def get_mask(self, idx):
        print(f"[DEBUG] get_mask called with idx: {idx}, current mask_index: {self.mask_index}")
        try:
            if idx < 0 or idx >= len(self.masks):
                print(f"[ERROR] Invalid Mask Index: {idx}, valid range is 0-{len(self.masks)-1}")
                raise StitchingError("Invalid Mask Index!")
            
            print(f"[DEBUG] Returning mask at index: {idx}")
            return self.masks[idx]
            
        except IndexError:
            print("[ERROR] Ran out of masks!")
            raise StitchingError("Ran out of masks!")


    def blend_images(self, images, seam_est_masks, corners, sizes):
        try:
            # Initialize blender with corners and sizes
            if self.timelapser.do_timelapse:
                self.timelapser.initialize(corners, sizes)
            else:
                self.blender.prepare(corners, sizes)

            # Feed images to blender
            for idx, (img, mask, corner) in enumerate(zip(images, seam_est_masks, corners)):
                try:
                    # Convert UMat to regular numpy array if needed
                    if isinstance(img, cv2.UMat):
                        img = img.get()
                    if isinstance(mask, cv2.UMat):
                        mask = mask.get()
                        
                    print(f"Processing image {idx} with shape {img.shape}, mask shape {mask.shape}, corner {corner}")
                    
                    # Ensure img is of type CV_16SC3 and in valid range
                    if img.dtype != np.int16:
                        print(f"Converting image {idx} from {img.dtype} to int16")
                        # Scale values to appropriate range for int16
                        if img.dtype == np.uint8:
                            img = img.astype(np.int16) * 256
                        else:
                            img = img.astype(np.int16)
                    
                    # Verify image properties
                    if len(img.shape) != 3 or img.shape[2] != 3:
                        raise StitchingError(f"Image {idx} must have 3 channels, got shape {img.shape}")
                    
                    # Ensure mask is binary uint8
                    if mask.dtype != np.uint8:
                        print(f"Converting mask {idx} from {mask.dtype} to uint8")
                        mask = mask.astype(np.uint8)
                        
                    # Ensure mask is binary (0 or 255)
                    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
                    
                    print(f"Feeding blender with image {idx}: dtype={img.dtype}, shape={img.shape}, value range=[{img.min()}, {img.max()}]")
                    self.blender.feed(img, mask, corner)
                    
                except cv2.error as e:
                    raise StitchingError(f"OpenCV error while processing image {idx}: {str(e)}")
                except Exception as e:
                    raise StitchingError(f"Error processing image {idx}: {str(e)}")
            
        except Exception as e:
            raise StitchingError(f"Error during blending: {str(e)}")

    def create_final_panorama(self):
        if not self.timelapser.do_timelapse:
            try:
                print("Starting final blend...")
                result, result_mask = self.blender.blend()
                print(f"Blend complete. Result shape: {result.shape}, dtype: {result.dtype}")
                
                # Convert result to appropriate format if needed
                if result.dtype == np.int16:
                    # Scale back to uint8 range
                    result = np.clip(result / 256, 0, 255).astype(np.uint8)
                
                return result
            except cv2.error as e:
                raise StitchingError(f"Error in final blending: {str(e)}")
            except Exception as e:
                raise StitchingError(f"Unexpected error in final blending: {str(e)}")

    def validate_kwargs(self, kwargs):
        for arg in kwargs:
            if arg not in self.DEFAULT_SETTINGS:
                raise StitchingError("Invalid Argument: " + arg)


class AffineStitcher(Stitcher):
    AFFINE_DEFAULTS = {
        "estimator": "affine",
        "wave_correct_kind": "no",
        "matcher_type": "affine",
        "adjuster": "affine",
        "warper_type": "affine",
        "compensator": "no",
    }

    DEFAULT_SETTINGS = Stitcher.DEFAULT_SETTINGS.copy()
    DEFAULT_SETTINGS.update(AFFINE_DEFAULTS)

    def initialize_stitcher(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.AFFINE_DEFAULTS and value != self.AFFINE_DEFAULTS[key]:
                warnings.warn(
                    f"You are overwriting an affine default ({key}={self.AFFINE_DEFAULTS[key]}) with another value ({value}). Make sure this is intended",  # noqa: E501
                    StitchingWarning,
                )
        super().initialize_stitcher(**kwargs)