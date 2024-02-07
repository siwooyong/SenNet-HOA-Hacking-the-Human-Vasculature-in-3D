# SenNet-HOA-Hacking-the-Human-Vasculature-in-3D

# TLDR
The key factors in enhancing the model's robust performance included **Tversky loss**, **increased inference size**, and **resolution augmentation**. These elements ultimately played a significant role in the model's survival during shakeups. The final model is an ensemble composed of 2D U-Net models based on the RegNetY-016 architecture

# Interesting Point
It was a truly challenging competition to secure a reliable validation set (which, unfortunately, I couldn't achieve). Upon reviewing the results, I discovered a significant difference between the public and private sets. Surprisingly, there were submissions from the past that would have made it into the gold zone. It's astonishing, considering I didn't think much of those submissions and didn't end up submitting them anyway.![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8251891%2Fe5a2fef330678d164bc8ff54d06c3e34%2F2024-02-07%2021-58-54.png?generation=1707310804960891&alt=media)

In my experiments, I found that giving a higher weight to the positive class in the Tversky loss (with a larger beta) resulted in better performance on the private leaderboard. However, for my final submission, where I trained with a smaller beta value in the Tversky loss, the model performed well on the public leaderboard but surprisingly poorly on the private leaderboard. **This discrepancy may be attributed to the higher resolution of the private dataset, which likely contained more detailed input information. Consequently, it suggests that a lower threshold for the model logits might have been more appropriate in this context. Furthermore, scaling the image (1.2, 1.5, ...) during inference seemed to dilute input information, reducing the impact of resolution.**

# Data Processing
Three consecutive slices were used as input for the model, and training was performed with the original image size. Despite experimenting with an increased number of slices (5, 7, 9...), the results revealed a decrease in performance.

In the early stages of the competition, I trained the model by resizing the data to a specific size. However, as I mentioned in the [discussion](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/463121), the Resize function in the Albumentations library applies nearest interpolation to masks, causing significant noise in fine labels and resulting in a notable decrease in performance. Therefore, I opted to use the original image size.

# Augmentation
Awaring of the different resolutions between public and private data, I aimed to create a model robust to resolution variations. Additionally, understanding the existence of resizing during the binning process, I employed blur augmentation. 

    def blur_augmentation(x):
        h, w, _ = x.shape
        scale = np.random.uniform(0.5, 1.5)

        x = A.Resize(int(h*scale), int(w*scale))(image=x)['image']
        x = A.Resize(h, w)(image=x)['image']
        return x

Moreover, since the channels were constructed by stacking depth, I applied the following augmentations.

    def channel_augmentation(x, prob=0.5, n_channel=3):
        assert x.shape[2]==n_channel
        if np.random.rand()<prob:
            x = np.flip(x, axis=2)
        return x

Finally, given the inherent noise in the data annotations, I implemented strong cutout augmentation to prevent overfitting.

    A.Cutout(num_holes=8, max_h_size=128, max_w_size=128, p=0.8),

These approaches effectively contributed to performance improvement in both CV and LB.


# Model
The model, like most others, employed a 2D U-Net architecture. For the CNN backbone, I utilized the lightweight model RegNetY-016. Other than that, the settings remained consistent with the default values in SMP (Segmentation Models PyTorch).

Despite investing significant time in developing a 3D-based model, it failed to demonstrate notable score improvements in both CV and LB. Due to resource constraints, I shifted my focus to a 2D model.

    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()

            self.n_classes = 1
            self.in_chans = 3

            self.encoder = timm.create_model(
                'regnety_016',
                pretrained=True,
                features_only=True,
                in_chans=self.in_chans,
            )
            encoder_channels = tuple(
                [self.in_chans]
                + [
                    self.encoder.feature_info[i]["num_chs"]
                    for i in range(len(self.encoder.feature_info))
                ]
            )
            self.decoder = UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=(256, 128, 64, 32, 16),
                n_blocks=5,
                use_batchnorm=True,
                center=False,
                attention_type=None,
            )

            self.segmentation_head = SegmentationHead(
                in_channels=16,
                out_channels=self.n_classes,
                activation=None,
                kernel_size=3,
            )

            self.train_loss = smp.losses.TverskyLoss(mode='binary', alpha=0.1, beta=0.9)
            self.test_loss = smp.losses.DiceLoss(mode='binary')


        def forward(self, batch, training=False):

            x_in = batch["input"]

            enc_out = self.encoder(x_in)

            decoder_out = self.decoder(*[x_in] + enc_out)
            x_seg = self.segmentation_head(decoder_out)

            output = {}
            one_hot_mask = batch["mask"][:, None]
            if training:
                loss = self.train_loss(x_seg, one_hot_mask.float())
            else:
                loss = self.test_loss(x_seg, one_hot_mask.float())

            output["loss"] = loss
            output['logit'] = nn.Sigmoid()(x_seg)[:, 0]

            return output

# Train

* Scheduler : lr_warmup_cosine_decay 
* Warmup Ratio : 0.1
* Optimizer : AdamW 
* Weight Decay : 0.01
* Epoch : 20
* Learning Rate : 2e-4
* Loss Function : TverskyLoss(mode='binary', alpha=0.1, beta=0.9)
* Batchsize : 4
* Gradient Accumulation : 4

# Inference
Scaling the image size by 1.5x during inference consistently resulted in score improvements in CV, public LB, and private LB. This acted as a form of dilation, significantly reducing false negatives and enhancing the model's performance. **Simply increasing the image size by 1.2 for inference resulted in a 0.1 improvement on the private leaderboard.**

# Didn't Work
* I attempted to enhance the utility of kidney2 and kidney3 through pseudo-labeling, but it did not result in significant score improvement.
* Efforts to create a more robust model using heavier augmentations did not yield substantial effects.
* Experimenting with larger CNN models led to issues of overfitting.
