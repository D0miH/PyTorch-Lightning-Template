import hashlib
import os

from typing import Optional, Callable
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, check_integrity

from datasets.dataset_interface import DatasetInterface


class ImageNette(ImageFolder, DatasetInterface):
    """`ImageNette <https://github.com/fastai/imagenette>`_ Dataset
    Inspired by CIFAR10 class of TorchVision.
    """
    base_folder_160 = 'imagenette2-160'
    base_folder_320 = 'imagenette2-320'
    url_160 = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
    url_320 = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
    filename_160 = 'imagenette2-160.tgz'
    filename_320 = 'imagenette2-320.tgz'
    tgz_md5_160 = 'e793b78cc4c9e9a4ccc0c1155377a412'
    tgz_md5_320 = '3df6f0d01a2c9592104656642f5e78a3'
    extracted_folder_md5_160 = '29250c4ba2f6f3322372ff7a1b5ab99d'
    extracted_folder_md5_320 = '52be8ad908708ed40b424b25d0d0e738'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        size=160
    ):
        self.root = root
        self.train = train
        self.size = size

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        if size == 160:
            image_folder_path = os.path.join(root, self.base_folder_160, 'train' if self.train else 'val')
        elif size == 320:
            image_folder_path = os.path.join(root, self.base_folder_320, 'train' if self.train else 'val')
        else:
            raise RuntimeError(f'Given image size of {size} is not supported. Please user either 160 or 320.')
        super(ImageNette, self).__init__(image_folder_path, transform=transform, target_transform=target_transform)

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _check_integrity(self):
        if self.size == 160:
            # check the integrity of the zip
            if not check_integrity(os.path.join(self.root, self.filename_160)):
                return False

            # check the integrity of the extracted image folder
            if not self._check_folder_integrity(
                os.path.join(self.root, self.base_folder_160), self.extracted_folder_md5_160
            ):
                return False
        elif self.size == 320:
            # check the integrity of the zip
            if not check_integrity(os.path.join(self.root, self.filename_320)):
                return False

            # check the integrity of the extracted image folder
            if not self._check_folder_integrity(
                os.path.join(self.root, self.base_folder_320), self.extracted_folder_md5_320
            ):
                return False

        return True

    def _check_folder_integrity(self, folder_name, md5):
        """
        Inspired by https://stackoverflow.com/a/24937710. Special thanks to
        Andy <https://stackoverflow.com/users/189134/andy>.
        """
        md5sum = hashlib.md5()
        if not os.path.exists(folder_name):
            return False

        for root, dirs, files in os.walk(folder_name):
            dirs.sort()
            for fnames in sorted(files):
                fpath = os.path.join(root, fnames)
                try:
                    f = open(fpath, 'rb')
                except:
                    # if the file cannot be opened just continue
                    f.close()
                    continue

                for chunk in iter(lambda: f.read(4096), b''):
                    md5sum.update(hashlib.md5(chunk).digest())
                f.close()

        return md5sum.hexdigest() == md5

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        if self.size == 160:
            download_and_extract_archive(self.url_160, self.root, filename=self.filename_160, md5=self.tgz_md5_160)
        elif self.size == 320:
            download_and_extract_archive(self.url_320, self.root, filename=self.filename_320, md5=self.tgz_md5_320)
        else:
            raise RuntimeError(f'Given image size of {self.size} is not supported. Please user either 160 or 320.')