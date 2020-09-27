import argparse

from .models import *  # set ONNX_EXPORT in models.py
from .utils.datasets import *
from .utils.utils import *
from typing import List


# x0y0 - left down
# x1y1 - up right
class BBox:
    def __init__(self, x0, y0, x1, y1, class_index, confidence: float):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.class_index = class_index
        self.confidence = confidence


def create_model(cfg, weights, imgsz, half=False, device="cpu") -> Darknet:
    # increase speed? idk
    torch.backends.cudnn.benchmark = True

    # Initialize
    device = torch_utils.select_device(device)

    # Initialize model
    model = Darknet(cfg, imgsz)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    return model


def detect(model, img0, img_size, half=False, device="cpu", conf_thres=0.3, iou_thres=0.6, augment=True) -> List[BBox]:
    # convert frame to network friendly format
    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=augment)[0]
    t2 = torch_utils.time_synchronized()

    # to float
    if half:
        pred = pred.float()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres,
                               multi_label=False, classes=None, agnostic=False)

    boxes = []
    # Process detections
    for i, det in enumerate(pred):  # detections for image i
        if det is not None and len(det):
            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                boxes.append(BBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3], int(cls), conf))

    print(f"inference time: {t2 - t1:.2}s")
    return boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # opt = parser.parse_args()
    # opt.cfg = check_file(opt.cfg)  # check file
    # opt.names = check_file(opt.names)  # check file
    # print(opt)

    # with torch.no_grad():
    # detect()
