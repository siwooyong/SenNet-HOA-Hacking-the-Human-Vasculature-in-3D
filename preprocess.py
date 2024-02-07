# 3fold : per donor

def preprocess():
    train_rles = pd.read_csv('/content/train_rles.csv')

    datasets = []
    slices = []
    groups = []
    for i in range(len(train_rles)):
        id, rle = train_rles.loc[i]

        dataset_id = '_'.join(id.split('_')[:-1])
        slice_id = id.split('_')[-1]
        group_id = '_'.join(id.split('_')[:-1])

        if dataset_id == 'kidney_3_dense':
            dataset_id = 'kidney_3_sparse'

        datasets.append(dataset_id)
        slices.append(slice_id)
        groups.append(group_id)

    train_rles['dataset'] = datasets
    train_rles['slice'] = slices
    train_rles['group'] = groups

    folds = []
    for i in range(3):
        train_df = train_rles[~train_rles['dataset'].str.contains(f'{i+1}')].reset_index(drop=True)
        val_df = train_rles[train_rles['dataset'].str.contains(f'{i+1}')].reset_index(drop=True)
        folds.append([train_df, val_df])

    return train_rles, folds

if __name__ == "__main__":
    train_rles, folds = preprocess()
