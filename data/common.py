from data.augmentation import *
def get_aug(cfg, image_size, train,mean_std):

    if cfg.trn_phase == 'pretrain' and train:
        if cfg.arch == 'SimSiam':
            augmentation = SimSiamTransform(image_size,mean_std=mean_std)
        elif cfg.arch == 'byol':
            augmentation = BYOL_transform(image_size,mean_std=mean_std)
        elif cfg.arch == 'SimCLR':
            augmentation = SimCLRTransform(image_size,mean_std=mean_std)
        else:
            raise NotImplementedError
    # elif cfg.trn_phase == 'classify' and train:
    #     augmentation = Transform_single(image_size, train=train_classifier)
    else:
        augmentation = Transform_single(cfg,image_size, train=train,mean_std=mean_std)

    return augmentation
