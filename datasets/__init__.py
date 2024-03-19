from torch.utils.data import DataLoader

from .textile_dataset import TextileDataset


def load_dataset(args, is_train):
    dataset = TextileDataset(**vars(args), is_train=is_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size if is_train else 1,
                            shuffle=is_train, num_workers=args.num_workers, pin_memory=True)
    return dataloader


