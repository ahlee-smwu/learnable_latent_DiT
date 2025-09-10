import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LSUNBase(Dataset):
    def __init__(self,
                 txt_file=None,
                 data_root=None,
                 npy_file=None,
                 size=128, # vavae train: 128
                 interpolation="bicubic",
                 flip_p=0.5
                 ):

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        # npy 파일이 있으면 npy 사용
        self.npy_file = npy_file    # (N, H, W, C)
        if self.npy_file is not None:
            self.images = np.load(self.npy_file, mmap_mode='r')  # lazy loading
            self._length = len(self.images)
            self.labels = {"relative_file_path_": [str(i) for i in range(self._length)],
                           "file_path_": [None] * self._length}  # 기존 구조 호환용
        else:
            # 기존 txt + data_root 방식
            self.data_paths = txt_file
            self.data_root = data_root
            with open(self.data_paths, "r") as f:
                self.image_paths = f.read().splitlines()
            self._length = len(self.image_paths)
            self.labels = {
                "relative_file_path_": [l for l in self.image_paths],
                "file_path_": [os.path.join(self.data_root, l)
                               for l in self.image_paths],
            }

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        # npy 사용 시
        if self.npy_file is not None:
            img = self.images[i]
            example.pop("file_path_", None)
        else:
            image = Image.open(example["file_path_"])
            if not image.mode == "RGB":
                image = image.convert("RGB")
            img = np.array(image).astype(np.uint8)
        if img is None:
            raise ValueError(f"Failed to load image idx={i}")

        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        if self.size is not None:
            img = Image.fromarray(img)
            img = img.resize((self.size, self.size), resample=self.interpolation)
            img = np.array(img)

        img = self.flip(Image.fromarray(img))
        img = np.array(img).astype(np.uint8)

        import torch
        example["image"] = torch.from_numpy((img / 127.5 - 1.0).astype(np.float32)) # .permute(2, 0, 1)
        return example


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
                         flip_p=flip_p, **kwargs)


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)
