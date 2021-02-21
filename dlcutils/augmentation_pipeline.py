import imgaug.augmenters as iaa
import configparser


class PipelineCreator:
    def __init__(self, config_file):
        self.config_file = config_file
        self.augmentation_cfg = configparser.ConfigParser()
        self.augmentation_cfg.read(self.config_file)
        self.pipeline = None
        self.sometimes = None

    def initPipeline(self, apply_prob):
        # sometimes only applies a given augmentation with a certain probability.
        self.sometimes = lambda aug: iaa.Sometimes(apply_prob,
                                                   aug)
        # This is used to prevent heavy augmentation all the time
        self.pipeline = iaa.Sequential(random_order=False)

    def applyFlip_left_right(self):
        if self.augmentation_cfg.get('fliplr', 'value') == 'True':
            prob = float(self.augmentation_cfg.get('fliplr', 'prob'))
            self.pipeline.add(self.sometimes(iaa.Fliplr(p=prob)))

    def applyFlip_up_down(self):
        if self.augmentation_cfg.get('flipud', 'value') == 'True':
            prob = float(self.augmentation_cfg.get('flipud', 'prob'))
            self.pipeline.add(self.sometimes(iaa.Flipud(p=prob)))

    def applyRotation(self):
        if self.augmentation_cfg.get('rotation', 'value') == 'True':
            deg = int(self.augmentation_cfg.get('rotation', 'deg'))
            self.pipeline.add(self.sometimes(iaa.Affine(rotate=(-deg, deg))))

    def applyMotionblur(self):
        if self.augmentation_cfg.get('motionblur', 'value') == 'True':
            k = int(self.augmentation_cfg.get('motionblur', 'k'))
            angle = int(self.augmentation_cfg.get('motionblur', 'angle'))
            self.pipeline.add(self.sometimes(iaa.MotionBlur(k=k, angle=(-angle, angle))))

    def applyResize(self):
        if self.augmentation_cfg.get('resize', 'value') == 'True':
            width = int(self.augmentation_cfg.get('resize', 'width'))
            height = int(self.augmentation_cfg.get('resize', 'height'))
            self.pipeline.add(self.sometimes(iaa.Resize({"height": height, "width": width})))
            # iaa.Resize({"height": 32, "width": 64})

    def applyRegiondropout(self):
        if self.augmentation_cfg.get('regiondropout', 'value') == 'True':
            prob = float(self.augmentation_cfg.get('regiondropout', 'prob'))
            size_percent = float(self.augmentation_cfg.get('regiondropout', 'size_percent'))
            per_channel = float(self.augmentation_cfg.get('regiondropout', 'per_channel'))
            self.pipeline.add(
                self.sometimes(iaa.CoarseDropout(p=prob, size_percent=size_percent, per_channel=per_channel)))

    def applyElastictransform(self):
        if self.augmentation_cfg.get('elastictransform', 'value') == 'True':
            sigma = int(self.augmentation_cfg.get('elastictransform', 'sigma'))
            self.pipeline.add(self.sometimes(iaa.ElasticTransformation(sigma=sigma)))

    def applyGaussianNoise(self):
        if self.augmentation_cfg.get('gaussian_noise', 'value') == 'True':
            loc = float(self.augmentation_cfg.get('gaussian_noise', 'loc'))
            scalex = float(self.augmentation_cfg.get('gaussian_noise', 'scalex'))
            scaley = float(self.augmentation_cfg.get('gaussian_noise', 'scaley'))
            per_channel = float(self.augmentation_cfg.get('gaussian_noise', 'per_channel'))
            self.pipeline.add(
                self.sometimes(iaa.AdditiveGaussianNoise(loc=loc, scale=(scalex, scaley), per_channel=per_channel)))

    def applyMultiplyBrightness(self):
        if self.augmentation_cfg.get('multiply_brightness', 'value') == 'True':
            self.pipeline.add(self.sometimes(iaa.MultiplyBrightness((0.5, 1.5))))

    def applyGrayscale(self):
        if self.augmentation_cfg.get('grayscale', 'value') == 'True':
            alphax = float(self.augmentation_cfg.get('grayscale', 'alphax'))
            alphay = float(self.augmentation_cfg.get('grayscale', 'alphay'))
            self.pipeline.add(self.sometimes(iaa.Grayscale(alpha=(alphax, alphay))))

    def applyHistogramEqualization(self):
        if self.augmentation_cfg.get('hist_eq', 'value') == 'True':
            self.pipeline.add(self.sometimes(iaa.AllChannelsHistogramEqualization()))

    def applyCropping(self):
        if self.height is not None and self.width is not None:
            if self.augmentation_cfg.get('crop_by', 'value') == 'True':
                percent = float(self.augmentation_cfg.get('crop_by', 'percent'))
                keep_size = self.augmentation_cfg.get('crop_by', 'keep_size')
                cropratio = float(self.augmentation_cfg.get('crop_by', 'cropratio'))
                if keep_size == 'False':
                    keep_size = False
                else:
                    keep_size = True
                self.pipeline.add(
                    iaa.Sometimes(cropratio, iaa.CropAndPad(percent=(-percent, percent), keep_size=keep_size)))
                self.pipeline.add(iaa.Resize({"height": self.height, "width": self.width}))

    def applyPad(self):
        if self.augmentation_cfg.get('pad', 'value') == 'True':
            pad_width = int(self.augmentation_cfg.get('pad', 'padded_width'))
            pad_height = int(self.augmentation_cfg.get('pad', 'padded_height'))
            self.pipeline.add(iaa.PadToFixedSize(pad_width, pad_height, position="center"))

    def addAugmentationToPipeline(self, augmentationprob):
        self.initPipeline(augmentationprob)

        self.applyPad()
        self.applyFlip_left_right()
        self.applyFlip_up_down()
        self.applyRotation()
        self.applyMotionblur()
        self.applyRegiondropout()
        self.applyElastictransform()
        self.applyGaussianNoise()
        self.applyMultiplyBrightness()
        self.applyGrayscale()
        self.applyHistogramEqualization()
        self.applyResize()

    def getPipeline(self):
        return self.pipeline
