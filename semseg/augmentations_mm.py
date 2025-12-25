import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional
from torchvision import io



class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: list) -> list:
        img, mask = sample['img'], sample['mask']
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            sample = transform(sample)

        return sample


class Normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample: list) -> list:
        for k, v in sample.items():
            if k == 'mask':
                continue
            elif k == 'img':
                sample[k] = sample[k].float()
                sample[k] /= 255
                sample[k] = TF.normalize(sample[k], self.mean, self.std)
            else:
                sample[k] = sample[k].float()
                sample[k] /= 255
        
        return sample


class RandomColorJitter:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            self.brightness = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_brightness(sample['img'], self.brightness)
            self.contrast = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_contrast(sample['img'], self.contrast)
            self.saturation = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_saturation(sample['img'], self.saturation)
        return sample


class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.adjust_gamma(img, self.gamma, self.gain), mask


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.adjust_sharpness(sample['img'], self.sharpness)
        return sample


class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.autocontrast(sample['img'])
        return sample


class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.gaussian_blur(sample['img'], self.kernel_size)
            # img = TF.gaussian_blur(img, self.kernel_size)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            for k, v in sample.items():
                sample[k] = TF.hflip(v)
            return sample
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, 3)
        return img, mask


class Equalize:
    def __call__(self, image, label):
        return TF.equalize(image), label


class Posterize:
    def __init__(self, bits=2):
        self.bits = bits # 0-8
        
    def __call__(self, image, label):
        return TF.posterize(image, self.bits), label


class Affine:
    def __init__(self, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0], seg_fill=0):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.seg_fill = seg_fill
        
    def __call__(self, img, label):
        return TF.affine(img, self.angle, self.translate, self.scale, self.shear, TF.InterpolationMode.BILINEAR, 0), TF.affine(label, self.angle, self.translate, self.scale, self.shear, TF.InterpolationMode.NEAREST, self.seg_fill) 


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 0, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag. 
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
                else:
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
            # img = TF.rotate(img, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
            # mask = TF.rotate(mask, random_angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
        return sample
    

class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(img, self.size), TF.center_crop(mask, self.size)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h+1)
            x1 = random.randint(0, margin_w+1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]
        return img, mask


class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int], seg_fill: int = 0) -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            size: expected output image size (h, w)
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        padding = (0, 0, self.size[1]-img.shape[2], self.size[0]-img.shape[1])
        return TF.pad(img, padding), TF.pad(mask, padding, self.seg_fill)


# class ResizePad:
#     def __init__(self, size: Union[int, Tuple[int], List[int]], seg_fill: int = 0) -> None:
#         """Resize the input image to the given size.
#         Args:
#             size: Desired output size.
#                 If size is a sequence, the output size will be matched to this.
#                 If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
#         """
#         self.size = size
#         self.seg_fill = seg_fill
#
#     def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
#         H, W = img.shape[1:]
#         tH, tW = self.size
#
#         # scale the image
#         scale_factor = min(tH/H, tW/W) if W > H else max(tH/H, tW/W)
#         # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
#         nH, nW = round(H*scale_factor), round(W*scale_factor)
#         img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
#         mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)
#
#         # pad the image
#         padding = [0, 0, tW - nW, tH - nH]
#         img = TF.pad(img, padding, fill=0)
#         mask = TF.pad(mask, padding, fill=self.seg_fill)
#         return img, mask


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, sample:list) -> list:
        H, W = sample['img'].shape[1:]

        # scale the image 
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
        # img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        # mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.BILINEAR)
        # img = TF.resize(img, (alignH, alignW), TF.InterpolationMode.BILINEAR)
        # mask = TF.resize(mask, (alignH, alignW), TF.InterpolationMode.NEAREST)
        return sample


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        # img, mask = sample['img'], sample['mask']
        H, W = sample['img'].shape[1:]
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)
        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)

        # random crop
        margin_h = max(sample['img'].shape[1] - tH, 0)
        margin_w = max(sample['img'].shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]

        # pad the image
        if sample['img'].shape[1:] != self.size:
            padding = [0, 0, tW - sample['img'].shape[2], tH - sample['img'].shape[1]]
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.pad(v, padding, fill=self.seg_fill)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)

        return sample

class ResizedPad1:
    def __init__(self, size: Union[int, Tuple[int], List[int]],  seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        tH, tW = self.size

        # pad the image；
        if sample['img'].shape[1:] != self.size:
            padding = [0, 0, tW - sample['img'].shape[2], tH - sample['img'].shape[1]]
            for k, v in sample.items():
                if k == 'mask':
                    sample[k] = TF.pad(v, padding, fill=self.seg_fill)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)

        return sample


class SyncGeomAugment:
    def __init__(self, size: Union[int, Tuple[int, int], List[int]], p_flip: float = 0.5, max_rotate_deg: float = 10.0, seg_fill: int = 0, p_rotate: float = 0.5) -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.p_flip = p_flip
        self.max_rotate_deg = max_rotate_deg
        self.seg_fill = seg_fill
        self.p_rotate = p_rotate

    def _maybe_flip(self, sample: list) -> list:
        if random.random() < self.p_flip:
            for k, v in sample.items():
                sample[k] = TF.hflip(v)
        return sample

    def _maybe_rotate(self, sample: list) -> list:
        if random.random() < self.p_rotate:
            angle = random.random() * 2 * self.max_rotate_deg - self.max_rotate_deg
            for k, v in sample.items():
                if k == 'mask':
                    sample[k] = TF.rotate(v, angle, TF.InterpolationMode.NEAREST, expand=False, fill=self.seg_fill)
                else:
                    sample[k] = TF.rotate(v, angle, TF.InterpolationMode.BILINEAR, expand=False, fill=0)
        return sample

    def _crop_to_size(self, sample: list) -> list:
        tH, tW = self.size
        curH, curW = sample['img'].shape[1:]
        pad_h = max(tH - curH, 0)
        pad_w = max(tW - curW, 0)
        if pad_h > 0 or pad_w > 0:
            padding = [0, 0, pad_w, pad_h]
            for k, v in sample.items():
                if k == 'mask':
                    sample[k] = TF.pad(v, padding, fill=self.seg_fill)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)
            curH, curW = sample['img'].shape[1:]

        margin_h = max(curH - tH, 0)
        margin_w = max(curW - tW, 0)
        y1 = random.randint(0, margin_h) if margin_h > 0 else 0
        x1 = random.randint(0, margin_w) if margin_w > 0 else 0
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]
        return sample

    def __call__(self, sample: list) -> list:
        sample = self._maybe_flip(sample)
        sample = self._maybe_rotate(sample)
        sample = self._crop_to_size(sample)
        return sample


class RGBPhotometric:
    def __init__(self, p_color: float = 0.8, brightness: Tuple[float, float] = (0.2, 0.4), contrast: Tuple[float, float] = (0.2, 0.4),
                 saturation: Tuple[float, float] = (0.2, 0.4), hue: Tuple[float, float] = (0.02, 0.05), p_gamma: float = 0.3,
                 gamma_range: Tuple[float, float] = (0.9, 1.1), p_jpeg: float = 0.2, jpeg_q: Tuple[int, int] = (70, 100)) -> None:
        self.p_color = p_color
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_jpeg = p_jpeg
        self.jpeg_q = jpeg_q

    @staticmethod
    def _to_float01(img: Tensor) -> Tensor:
        if img.dtype == torch.uint8:
            return img.float() / 255.0
        return img

    @staticmethod
    def _to_uint8(img: Tensor) -> Tensor:
        if img.dtype != torch.uint8:
            return (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return img

    def _apply_color_jitter(self, img: Tensor) -> Tensor:
        imgf = self._to_float01(img)
        if random.random() < self.p_color:
            b = 1.0 + random.uniform(-self.brightness[1], -self.brightness[0]) if random.random() < 0.5 else 1.0 + random.uniform(self.brightness[0], self.brightness[1])
            c = 1.0 + random.uniform(-self.contrast[1], -self.contrast[0]) if random.random() < 0.5 else 1.0 + random.uniform(self.contrast[0], self.contrast[1])
            s = 1.0 + random.uniform(-self.saturation[1], -self.saturation[0]) if random.random() < 0.5 else 1.0 + random.uniform(self.saturation[0], self.saturation[1])
            h = random.uniform(-self.hue[1], -self.hue[0]) if random.random() < 0.5 else random.uniform(self.hue[0], self.hue[1])
            imgf = TF.adjust_brightness(imgf, b)
            imgf = TF.adjust_contrast(imgf, c)
            imgf = TF.adjust_saturation(imgf, s)
            imgf = TF.adjust_hue(imgf, h)
        return self._to_uint8(imgf)

    def _apply_gamma(self, img: Tensor) -> Tensor:
        if random.random() < self.p_gamma:
            g = random.uniform(self.gamma_range[0], self.gamma_range[1])
            imgf = self._to_float01(img)
            imgf = TF.adjust_gamma(imgf, g)
            return self._to_uint8(imgf)
        return img

    def _apply_jpeg(self, img: Tensor) -> Tensor:
        if random.random() < self.p_jpeg:
            try:
                q = random.randint(self.jpeg_q[0], self.jpeg_q[1])
                img_u8 = img if img.dtype == torch.uint8 else self._to_uint8(img)
                # encode_jpeg/decoder 仅支持 CPU uint8 CHW/HWC
                img_cpu = img_u8.cpu()
                bytes_jpg = io.encode_jpeg(img_cpu, quality=q)
                img_dec = io.decode_jpeg(bytes_jpg)  # 返回 uint8，形状依版本可能为 CHW 或 HWC
                if img_dec.dim() == 3:
                    if img_dec.shape[0] in (1, 3):
                        out = img_dec
                    elif img_dec.shape[-1] in (1, 3):
                        out = img_dec.permute(2, 0, 1).contiguous()
                    else:
                        return img  # 异常形状，回退
                else:
                    return img
                return out.to(img.device)
            except Exception:
                return img
        return img

    def __call__(self, sample: list) -> list:
        if 'img' in sample:
            x = sample['img']
            x = self._apply_color_jitter(x)
            x = self._apply_gamma(x)
            x = self._apply_jpeg(x)
            sample['img'] = x
        return sample


class ThermalPhotometric:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _to_float255(img: Tensor) -> Tensor:
        return img.float() if img.dtype == torch.uint8 else img * 255.0

    @staticmethod
    def _from_float255(img: Tensor, like: Tensor) -> Tensor:
        if like.dtype == torch.uint8:
            return img.clamp(0.0, 255.0).round().to(torch.uint8)
        else:
            return (img.clamp(0.0, 255.0) / 255.0).to(like.dtype)

    def _gaussian_noise(self, img: Tensor) -> Tensor:
        img255 = self._to_float255(img)
        sigma = random.uniform(2.0, 6.0)
        noise = torch.randn_like(img255) * sigma
        out = img255 + noise
        return self._from_float255(out, img)

    def _poisson_noise(self, img: Tensor) -> Tensor:
        img01 = (img.float() / 255.0) if img.dtype == torch.uint8 else img.clamp(0.0, 1.0)
        scale = random.uniform(20.0, 40.0)
        lam = (img01 * scale).clamp(min=0.0)
        noisy = torch.poisson(lam) / scale
        out = noisy * 255.0
        return self._from_float255(out, img)

    def _gaussian_blur(self, img: Tensor) -> Tensor:
        sigma = random.uniform(0.4, 0.9)
        return TF.gaussian_blur(img, kernel_size=(3, 3), sigma=sigma)

    def __call__(self, sample: list) -> list:
        if 'thermal' in sample:
            t = sample['thermal']
            u = random.random()
            if u < 0.5:
                t = self._gaussian_noise(t)
            elif u < 0.8:
                t = self._poisson_noise(t)
            else:
                t = self._gaussian_blur(t)
            sample['thermal'] = t
        return sample

def get_train_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose([
        SyncGeomAugment(size=size, p_flip=0.5, max_rotate_deg=10.0, seg_fill=seg_fill, p_rotate=0.5),
        RGBPhotometric(p_color=0.8, brightness=(0.2, 0.4), contrast=(0.2, 0.4), saturation=(0.2, 0.4), hue=(0.02, 0.05),
                       p_gamma=0.3, gamma_range=(0.9, 1.1), p_jpeg=0.0, jpeg_q=(70, 100)),
        ThermalPhotometric(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def get_val_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose([
        ResizedPad1(size,  seg_fill=seg_fill), #
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


if __name__ == '__main__':
    h = 230
    w = 420
    sample = {}
    sample['img'] = torch.randn(3, h, w)
    sample['depth'] = torch.randn(3, h, w)
    sample['lidar'] = torch.randn(3, h, w)
    sample['event'] = torch.randn(3, h, w)
    sample['mask'] = torch.randn(1, h, w)
    aug = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop((512, 512)),
        Resize((224, 224)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    sample = aug(sample)
    for k, v in sample.items():
        print(k, v.shape)