import os


class UNetConfig:

    def __init__(self,
                 epochs=1000,
                 early_stop=30,
                 batch_size=8,
                 validation=0.1,  # Percent of the data that is used as validation (0-1)
                 out_threshold=0.5,

                 lr=0.0001,# learning rate
                 lr_decay_milestones=(20, 50),
                 lr_decay_gamma=0.9,
                 weight_decay=1e-8,
                 momentum=0.9,
                 nesterov=True,

                 n_channels=1,
                 n_classes=1,
                 scale=1,

                 load=False,

                 bilinear=True,
                 deepsupervision=True,
                 aug=True
                 ):
        super(UNetConfig, self).__init__()

        self.images_dir = '/home/test/hgb/data/Intestinal/img2D/images'
        self.masks_dir = '/home/test/hgb/data/Intestinal/img2D/masks'
        self.checkpoints_dir = './models'
        self.model = "NestedUNet"
        self.pretrain_model = None

        self.epochs = epochs
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold
        self.aug = aug
        self.loss = "BCEDiceLoss"

        self.optimizer = "Adam"
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale

        self.load = load

        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        os.makedirs(self.checkpoints_dir, exist_ok=True)
