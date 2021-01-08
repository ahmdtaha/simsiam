from PIL import Image, ImageOps
import torchvision.transforms as T
from torchvision.transforms import GaussianBlur

# imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
# cifar_mean_std = [[0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]]

# cifar_mean_std = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]

# imagenet_mean_std = [[0.0,0.0,0.0],[1,1,1]]


class SimSiamTransform():
    def __init__(self, image_size, mean_std):
        image_size = 224 if image_size is None else image_size  # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0
        # p_blur = 0.5
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

class SimCLRTransform():
    def __init__(self, image_size, mean_std, s=1.0):
        image_size = 224 if image_size is None else image_size
        p_blur = 0.5 if image_size > 32 else 0
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            # We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

class Solarization():
    # ImageFilter
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)

class BYOL_transform:  # Table 6
    def __init__(self, image_size, mean_std):
        self.transform1 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                                         interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0)),
            # simclr paper gives the kernel size. Kernel size has to be odd positive number with torchvision
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.transform2 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                                         interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * image_size))], p=0.1),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=0.1),
            T.RandomApply([Solarization()], p=0.2),

            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        x1 = self.transform1(x)
        x2 = self.transform2(x)
        return x1, x2

class Transform_single():
    def __init__(self, cfg ,image_size, train, mean_std):
        if train == True:
            # self.transform = T.Compose([
            #     # T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
            #     T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0)),
            #     T.RandomHorizontalFlip(),
            #     T.ToTensor(),
            #     T.Normalize(*normalize)
            # ])
            p_blur = 0.5
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
                T.ToTensor(),
                T.Normalize(*mean_std)
            ])
        else:
            if cfg.set in ['CIFAR10', 'CIFAR100'] :
                self.transform = T.Compose([
                    # T.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256
                    # T.Resize(int(image_size*(8/7))), # 224 -> 256
                    # T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize(*mean_std)
                ])
            elif cfg.set == 'ImageNet':
                self.transform = T.Compose([
                    T.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256
                    T.Resize(int(image_size*(8/7))), # 224 -> 256
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize(*mean_std)
                ])
            else:
                raise NotImplementedError('Invalid db_set {}'.format(cfg.set))
    def __call__(self, x):
        return self.transform(x)
