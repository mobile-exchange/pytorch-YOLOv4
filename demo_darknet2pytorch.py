from tool.torch_utils import device
from tool.darknet2pytorch import Darknet
import argparse
import torch


def convert2pytorch(cfgfile, weightfile, output):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    m = m.to(device())

    torch.save(m.state_dict(), output)


def get_args():
    parser = argparse.ArgumentParser(
        'Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4-tiny.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/yolov4-tiny.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-output', type=str,
                        default='./checkpoints/yolov4-tiny.pth',
                        help='path of trained model.', dest='output')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    convert2pytorch(args.cfgfile, args.weightfile, args.output)
