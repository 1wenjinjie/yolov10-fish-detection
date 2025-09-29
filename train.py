from ultralytics import YOLOv10
from ultralytics.utils.configfile import __train
import warnings
import argparse
warnings.filterwarnings('ignore')



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights',type=str, default='yolov10n.pt', help='loading pretrain weights')
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/improve/yolov10n-1.yaml', help='models') # 填写训练的网络模型名称
    parser.add_argument('--data', type=str, default='datasets.yaml', help='datasets')
    parser.add_argument('--epochs', type=int, default=300, help='train epoch')
    parser.add_argument('--batch', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=640, help='image sizes')
    parser.add_argument('--optimizer', default='SGD', help='use optimizer')#Adam
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    return parser.parse_args()

if __name__ == '__main__':
    args=main()
    # 开始加载模型
    model = YOLOv10(__train(args.cfg))
    if(".pt" in args.weights):
        print("+++++++载入预训练权重：",args.weights,"++++++++")
        model.load(args.weights)
    else:
        print("-------没有载入预训练权重-------")
    # 指定训练参数开始训练
    model.train(data=args.data,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                optimizer=args.optimizer,
                project=args.project,
                name=args.name)