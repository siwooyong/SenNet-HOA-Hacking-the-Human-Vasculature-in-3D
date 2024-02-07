class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.n_classes = 1#len(cfg.classes)
        in_chans = 3

        self.encoder = timm.create_model(
            'regnety_016',#cfg.backbone,
            pretrained=True,#cfg.pretrained,
            features_only=True,
            in_chans=in_chans,
        )
        encoder_channels = tuple(
            [in_chans]
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

        self.train_loss = smp.losses.TverskyLoss(mode='binary', alpha=0.1, beta=0.9)#DiceBCELoss()#smp.losses.DiceLoss(mode='binary')#nn.BCEWithLogitsLoss()
        self.test_loss = smp.losses.DiceLoss(mode='binary')#nn.BCEWithLogitsLoss()

        #self.return_logits = cfg.return_logits

    def forward(self, batch, training=False):

        x_in = batch["input"]

        enc_out = self.encoder(x_in)

        decoder_out = self.decoder(*[x_in] + enc_out)
        x_seg = self.segmentation_head(decoder_out)

        output = {}
        #if (not self.training) & self.return_logits:
        #    output["logits"] = x_seg

        #if self.training:
        #if self.n_classes > 1:
        #    one_hot_mask = F.one_hot(
        #        batch["mask"].long(), num_classes=self.n_classes + 1
        #    ).permute(0, 3, 1, 2)[:, 1:]
        #else:
        one_hot_mask = batch["mask"][:, None]
        if training:
            loss = self.train_loss(x_seg, one_hot_mask.float())
        else:
            loss = self.test_loss(x_seg, one_hot_mask.float())

        output["loss"] = loss
        output['logit'] = nn.Sigmoid()(x_seg)[:, 0]

        return output

if __name__ == "__main__":
    train_df, val_df = folds[2]
    ds = CustomDatasetV2(imgs, masks, 2, train_transform, training=True)
    loader = torch.utils.data.DataLoader(ds, batch_size = 1, num_workers = 8, shuffle = True, drop_last = True)
    sample = next(iter(loader))
    sample = [x.to(device) for x in sample]

    batch = {}
    batch['input'] = sample[0]
    batch['mask'] = sample[1]

    model = CustomModel().to(device)

    with torch.no_grad():
        output = model(batch, training=True)
        print(output)
