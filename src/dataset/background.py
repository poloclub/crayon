# Code from https://github.com/MadryLab/backgrounds_challenge/blob/master/tools/folder.py and https://github.com/MadryLab/backgrounds_challenge/blob/master/tools/datasets.py
import os 
import os.path
import sys
import numpy as np 
from PIL import Image 
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch




IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


BACKGROUND_TRANSFORM_TEST = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575]),
])

BACKGROUND_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    Lighting(0.05, IMAGENET_PCA['eigval'], IMAGENET_PCA['eigvec']),
    transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575]),
])
BACKGROUND_TRANSFORM_SEG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
# BACKGROUND_TRANSFORM_TRAIN = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
#     transforms.ToTensor(),
#     Lighting(0.05, IMAGENET_PCA['eigval'], IMAGENET_PCA['eigvec']),
#     transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575]),
# ])




def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None,
                 target_transform=None, label_mapping=None, seg_transform=None, reshape_seg=True):
        classes, class_to_idx = self._find_classes(root)
        if label_mapping is not None:
            classes, class_to_idx = label_mapping(classes, class_to_idx)

        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.E = None 

        self.transform = transform
        self.seg_transform = seg_transform
        self.target_transform = target_transform

        self.paths, self.labels = zip(*self.samples)
        self.labels = np.array(self.labels)

        self.box_provided = np.ones([45405])
        self.seg_provided = np.ones([45405])
        self.reshape_seg = reshape_seg

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        seg = None
        seg_path = path.split("/")
        seg_path[6] = "only_fg"
        seg_path = "/".join(seg_path)
        if os.path.exists(seg_path):
            seg = self.loader(seg_path) 
            
            box = np.array(seg)
            box_pos_idx = np.where(box>0)
            box_l, box_r, box_t, box_b = np.min(box_pos_idx[0]), np.max(box_pos_idx[0]), np.min(box_pos_idx[1]), np.max(box_pos_idx[1])
            box = np.zeros_like(box)
            box[box_l:box_r+1, box_t:box_b+1] = 255
            box = Image.fromarray(box)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.seg_transform is not None and seg is not None:
            seg = (torch.sum(self.seg_transform(seg), dim=0) != 0).float()
            box = (torch.sum(self.seg_transform(box), dim=0) != 0).float()
            if self.reshape_seg:
                seg = seg.reshape(7, 32, 7, 32).mean(axis=(1,3))
                box = box.reshape(7, 32, 7, 32).mean(axis=(1,3))
        if self.target_transform is not None:
            target = self.target_transform(target)
        expl = None
        if self.E is not None:
            expl = self.E[index]

        if expl is not None and seg is not None: 
            return {"idx": index, "img": sample, "img_name": path, "label": target, "path": path, "expl": expl, "seg": seg, "box": box, "seg_provided":self.seg_provided[index], "box_provided": self.box_provided[index]}
        elif seg is not None:
            return {"idx": index, "img": sample, "img_name": path, "label": target, "path": path, "seg": seg, "box": box, "seg_provided":self.seg_provided[index], "box_provided": self.box_provided[index]}
        elif expl is not None:
            return {"idx": index, "img": sample, "img_name": path, "label": target, "path": path, "expl": expl}
        else:
            return {"idx": index, "img": sample, "img_name": path, "label": target, "path": path}

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, label_mapping=None, seg_transform=None, reshape_seg=True):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          label_mapping=label_mapping, seg_transform=seg_transform, reshape_seg=reshape_seg)
        self.imgs = self.samples

class TensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        im, targ = tuple(tensor[index] for tensor in self.tensors)

        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            im = real_transform(im)

        return {"img": im, "label": targ}

    def __len__(self):
        return self.tensors[0].size(0)