import argparse

# TODO : default 값 적절한 값으로 변경 필요.
def par_args():
    parser = argparse.ArgumentParser(description='This network is for OpenImages2019')
    parser.add_argument('--dataset', dest='dataset',
                        type=str, default='../Detection/[target_dir/train]',
                        help='training dataset path')
    parser.add_argument('--mode', dest='mode',
                        type=str, default='train',
                        choices=['train', 'test'],
                        help='train or test')
    parser.add_argument('--ep', dest='epoch',
                        type=int, default=100,
                        help='train epoch')
    parser.add_argument('--lr', dest='lr',
                        type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--b', dest='batch',
                        type=int, default=128,
                        help='batch size')
    parser.add_argument('--weights', dest='weights',
                        type=str, default='weight.pt',
                        help='weight path')
    parser.add_argument('--output', dest='output',
                        type=str, default='output.csv',
                        help='output path')

    args = parser.parse_args()
    return args
