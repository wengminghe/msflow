import argparse

from train import Trainer


def main():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--class_name', type=str, default='fiber')
    parser.add_argument('--input_size', type=int, default=512, help='input image size')
    parser.add_argument('--random_mask', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')

    # model
    parser.add_argument('--extractor', type=str, default='wide_resnet50_2',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'wide_resnet50_2'])
    parser.add_argument('--pool_type', type=str, default='avg', help='pool type for extracted feature maps')
    parser.add_argument('--parallel_blocks', type=int, default=[2, 5, 8])
    parser.add_argument('--c_conds', type=int, default=[64, 64, 64])
    parser.add_argument('--clamp_alpha', type=float, default=1.9)

    # train
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--lr_milestones', type=int, default=[20], help='milestones')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='learning rate decay gamma')
    parser.add_argument('--output_dir', type=str, help='output directory')

    # eval
    parser.add_argument('--top_k', type=float, default=0.03)
    parser.add_argument('--threshold', type=float, default=0.8)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    main()

