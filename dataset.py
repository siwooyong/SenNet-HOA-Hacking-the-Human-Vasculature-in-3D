# 2d

'''train_transform = A.Compose(
    [
        A.Resize(int(img_size*1.125), int(img_size*1.125)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.5),
        A.RandomCrop(
            always_apply=False, p=1.0, height=img_size, width=img_size
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5
        ),
        A.Cutout(num_holes=8, max_h_size=int((img_size/512)*36), max_w_size=int((img_size/512)*36), p=0.8),
    ]
)'''

train_transform = A.Compose(
    [
        #A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Cutout(num_holes=8, max_h_size=128, max_w_size=128, p=0.8),
    ]
)

val_transform = A.Compose(
    [
        #A.Resize(img_size, img_size),
    ]
)

def blur_augmentation(x):
    h, w, _ = x.shape
    scale = np.random.uniform(0.5, 1.5)

    x = A.Resize(int(h*scale), int(w*scale))(image=x)['image']
    x = A.Resize(h, w)(image=x)['image']
    return x

def channel_augmentation(x, prob=0.5, n_channel=3):
    assert x.shape[2]==n_channel
    if np.random.rand()<prob:
        x = np.flip(x, axis=2)
    return x


class CustomDatasetV2(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, axis, transform, training=False):
        self.imgs = imgs
        self.masks = masks
        self.axis = axis
        self.transform = transform
        self.training = training

        self.n_slice = 1
        self.n_stride = 1

    def __len__(self):
        return self.imgs.shape[self.axis]

    def normalize_img(self, img):
        img = img - np.min(img)
        img = img / np.max(img)
        img = (img*255).astype(np.uint8)
        return img

    def process_img(self, index):
        img = []
        for i in range(-self.n_slice*self.n_stride, self.n_slice*self.n_stride+1, self.n_stride):
            i = i + index
            try:
                assert i >= 0 and i <= self.imgs.shape[self.axis]-1
                if self.axis==0:
                    x = self.imgs[i]
                elif self.axis==1:
                    x = self.imgs[:, i]
                elif self.axis==2:
                    x = self.imgs[:, :, i]
                else:
                    raise NotImplementedError()

                x = self.normalize_img(x)
                img.append(x)
            except:
                if self.axis==0:
                    x = np.zeros_like(self.imgs[0])
                elif self.axis==1:
                    x = np.zeros_like(self.imgs[:, 0])
                elif self.axis==2:
                    x = np.zeros_like(self.imgs[:, :, 0])
                else:
                    raise NotImplementedError()

                x = x.astype(np.uint8)
                img.append(x)

        img = np.stack(img, axis=-1)
        return img

    def process_mask(self, index):
        if self.axis==0:
            mask = self.masks[index]
        elif self.axis==1:
            mask = self.masks[:, index]
        elif self.axis==2:
            mask = self.masks[:, :, index]
        else:
            raise NotImplementedError()
        return mask

    def __getitem__(self, index):
        img = self.process_img(index)
        mask = self.process_mask(index)

        hw = torch.tensor(img.shape[:2])

        if self.training:
            img = blur_augmentation(img)
            #img = channel_augmentation(img)

        transforms = self.transform(image=img, mask=mask)
        img, mask = transforms['image'], transforms['mask']

        img = torch.tensor(img, dtype = torch.float)
        img = img.permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype = torch.float)

        return img, mask, hw

if __name__ == "__main__":

    train_df, val_df = folds[2]

    # train:kidney_1_dense
    train_df = train_df[train_df['group']=='kidney_1_dense'].reset_index(drop=True)
    #imgs, masks = preload(train_df)

    ds = CustomDatasetV2(imgs, masks, 1, train_transform, training=True)
    index = np.random.randint(0, len(ds)-1)

    img, mask, hw = ds[index]
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(img.permute(1, 2, 0), cmap='gray')
    axs[1].imshow(img[0], cmap='gray')
    axs[2].imshow(img[1], cmap='gray')
    axs[3].imshow(img[2], cmap='gray')
    axs[4].imshow(mask, cmap='gray')
    plt.show()
    print(hw)
