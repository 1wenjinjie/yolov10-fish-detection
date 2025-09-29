from ultralytics import YOLOv10
from ultralytics.utils.configfile import __test
import warnings
import argparse
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights',type=str, default='runs/train/exp/weights/best.pt', help='loading weights')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    return parser.parse_args()

if __name__ == '__main__':
    args=main()
    # 开始加载模型
    model = YOLOv10(__test(args.weights))
    # 指定训练参数开始验证
    model.val(device=args.device,
              project=args.project,
              name=args.name)
























