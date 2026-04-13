from jedepth.dataset.depth_dataset import StereoDepthDataset


def build_dataset(args, training=True):
    if training:
        csv_path = args.data_list_train[0]
        root = args.data_root_train[0]
        augment = bool(args.data_augmentation)
    else:
        csv_path = args.data_list_val[0]
        root = args.data_root_val[0]
        augment = False

    return StereoDepthDataset(
        csv_path=csv_path,
        root=root,
        augment=augment,
    )