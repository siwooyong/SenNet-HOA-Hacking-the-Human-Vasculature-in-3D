def train_function(model,
                   optimizer,
                   scheduler,
                   scaler,
                   loader,
                   device,
                   iters_to_accumulate):
    model.train()

    gc.collect()

    total_loss = 0.0
    for bi, sample in enumerate(tqdm(loader)):
        sample = [x.to(device) for x in sample]

        batch = {}

        batch['input'] = sample[0]
        batch['mask'] = sample[1]

        with torch.cuda.amp.autocast():
            loss = model(batch, training=True)['loss']
            loss = loss / iters_to_accumulate

        scaler.scale(loss).backward()
        if (bi + 1) % iters_to_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()

        total_loss += loss.detach().cpu() * iters_to_accumulate

    return total_loss/len(loader)

def val_function(model,
                 scaler,
                 loader,
                 device,
                 log_path,
                 threshold=0.5):
    model.eval()

    gc.collect()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    for bi, sample in enumerate(tqdm(loader)):
        sample = [x.to(device) for x in sample]

        batch = {}

        batch['input'] = sample[0]
        batch['mask'] = sample[1]

        with torch.no_grad():
            output = model(batch)
            loss = output['loss']
            logit = output['logit']

        dice = dice_score(logit>threshold, sample[1])
        iou = IoU_score(logit>threshold, sample[1])


        total_loss += loss.detach().cpu()
        total_dice += dice.detach().cpu()
        total_iou += iou.detach().cpu()

    message = {
        'bce_loss' : round(total_loss.tolist()/len(loader), 4),
        'dice_score' : round(total_dice.tolist()/len(loader), 4),
        'iou_score' : round(total_iou.tolist()/len(loader), 4)
    }

    with open(log_path, 'a+') as logger:
        logger.write(f'{message}\n')

    return message

try:del imgs, masks, ds, loader
except:pass

train, folds = preprocess()
train_df, val_df = folds[k]

for k in range(2, 3):

    batch_size = 4
    epoch = 20
    early_stop = 20
    lr = 2e-4
    wd = 0.01
    warmup_ratio = 0.1
    num_workers = 8
    iters_to_accumulate = 4
    train_dir = 'model:unet-regnety016,loss:tversky19,imgsize:original,train:kidney_1_dense,aug:flip+cutout+blur+xyz,channel:3'
    seed = 42
    root = '/content/drive/MyDrive/Kaggle/SenNet + HOA - Hacking the Human Vasculature in 3D/'

    seed_everything(seed)

    # train:kidney_1_dense
    train_df = train_df[train_df['dataset']=='kidney_1_dense'].reset_index(drop=True)

    train_imgs, train_masks = preload(train_df)
    val_imgs, val_masks = preload(val_df)

    train_dataset0 = CustomDatasetV2(train_imgs, train_masks, 2, train_transform, training=True)
    train_dataset1 = CustomDatasetV2(train_imgs, train_masks, 1, train_transform, training=True)
    train_dataset2 = CustomDatasetV2(train_imgs, train_masks, 0, train_transform, training=True)
    val_dataset = CustomDatasetV2(val_imgs, val_masks, 0, val_transform, training=False)#CustomDataset(val_df, val_transform)

    train_loader0 = torch.utils.data.DataLoader(train_dataset0, batch_size = batch_size, num_workers = num_workers, shuffle = True, drop_last = True)
    train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size = batch_size, num_workers = num_workers, shuffle = True, drop_last = True)
    train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size = batch_size, num_workers = num_workers, shuffle = True, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False, drop_last = False)

    model = CustomModel().to(device)

    optimizer = torch.optim.AdamW(params = model.parameters(), lr = lr, weight_decay = wd)
    total_steps = int((len(train_dataset0)+len(train_dataset1)+len(train_dataset2)) * epoch/(batch_size * iters_to_accumulate))
    warmup_steps = int(total_steps * warmup_ratio)
    print('total_steps: ', total_steps)
    print('warmup_steps: ', warmup_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps = warmup_steps,
                                                num_training_steps = total_steps)
    scaler = torch.cuda.amp.GradScaler()

    if not os.path.isdir(root + f'{train_dir}/'):
        os.mkdir(root + f'{train_dir}/')

    if not os.path.isdir(root + f'{train_dir}/fold{k+1}/'):
        os.mkdir(root + f'{train_dir}/fold{k+1}/')


    for i in range(epoch):
        # train0
        train_loss0 = train_function(model,
                                     optimizer,
                                     scheduler,
                                     scaler,
                                     train_loader0,
                                     device,
                                     iters_to_accumulate)
        # train1
        train_loss1 = train_function(model,
                                     optimizer,
                                     scheduler,
                                     scaler,
                                     train_loader1,
                                     device,
                                     iters_to_accumulate)
        # train2
        train_loss2 = train_function(model,
                                     optimizer,
                                     scheduler,
                                     scaler,
                                     train_loader2,
                                     device,
                                     iters_to_accumulate)

        train_loss = (train_loss0 + train_loss1 + train_loss2)/3
        # val
        message = val_function(model,
                               scaler,
                               val_loader,
                               device,
                               root + f'{train_dir}/fold{k+1}/log.txt')

        val_loss, val_dice, val_iou = message['bce_loss'], message['dice_score'], message['iou_score']


        # save
        save_path = root + f'{train_dir}/fold{k+1}/epoch' + f'{i+1}'.zfill(3) + \
                    f'-trainloss{round(train_loss.tolist(), 4)}' + \
                    f'-valloss{val_loss}' + \
                    f'-valdice{val_dice}' + \
                    f'-valiou{val_iou}' + '.bin'
        torch.save(model.state_dict(), save_path)

        _lr = optimizer.param_groups[0]['lr']
        print(f'epoch : {i+1}, lr : {_lr}, trainloss : {round(train_loss.tolist(), 4)}, valloss : {val_loss}, valdice : {val_dice}, valiou : {val_iou}')

        if i+1 == early_stop:
            break
