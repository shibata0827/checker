# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
run inference
"""

import cv2
from PIL import Image, ImageTk
import time
import os
import sys
from pathlib import Path
import datetime
import numpy as np
import yaml

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # main parent root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.datasets import LoadStreams
from models.common import DetectMultiBackend
from utils.general import check_img_size, cv2, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors, save_one_box

class Inference_app():
    def __init__(self, q, config_path, set_config_path, set_class_path, push_inference, finish_load, close_flg, show_results):
        self.q = q

        self.config_path = config_path
        with open(self.config_path, 'r', encoding='utf-8') as yml:
            self.config_data = yaml.load(yml, Loader=yaml.SafeLoader)
        self.weights=self.config_data["weights"]
        self.source=self.config_data["source"]
        self.imgsz=eval(self.config_data["imgsz"])
        self.conf_thres=self.config_data["conf_thres"]
        self.iou_thres=self.config_data["iou_thres"]
        self.save_txt=self.config_data["save_txt"]
        self.save_conf=self.config_data["save_conf"]
        self.save_crop=self.config_data["save_crop"]
        self.save_img=self.config_data["save_img"]
        self.project=self.config_data["project"]
        self.line_thickness=self.config_data["line_thickness"]
        self.hide_labels=self.config_data["hide_labels"]
        self.hide_conf=self.config_data["hide_conf"]
        self.save_evidence=self.config_data["save_evidence"]
        self.camera_rot=self.config_data["camera_rot"]
        self.evidence_project=self.config_data["evidence_project"]

        if not isinstance(self.save_evidence, bool):
            raise Exception(f'ERROR: self.save_evidence is not bool')
        if not self.camera_rot in [-90, 0, 90, 180]:
            raise Exception(f'ERROR: camera_rot not in [-90, 0, 90, 180]')

        self.set_config_path = set_config_path
        with open(self.set_config_path, 'r', encoding='utf-8') as yml:
            self.set_config_data = yaml.load(yml, Loader=yaml.SafeLoader)

        self.set_class_path = set_class_path
        with open(self.set_class_path, 'r', encoding='utf-8') as yml:
            self.set_class_data = yaml.load(yml, Loader=yaml.SafeLoader)

        self.results = []
        self.judgement = None
        self.push_inference = push_inference
        self.finish_load = finish_load
        self.close_flg = close_flg
        self.show_results = show_results
        self.s_time = 0

    def load_model_data(self):
        self.source = str(self.source)
        self.device = torch.device('cpu')

        # Load model
        self.model = DetectMultiBackend(self.weights, device=self.device)
        self.names = self.model.names
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride)  # check image size

        # Dataloader
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.model.stride,
                              auto=self.model.pt, camera_rot=self.camera_rot)

    def observe_flg(self):
        """
        フラグ監視と初期化
        """
        # 開始ボタンを押していない、OKNG判定結果を画面に表示している
        if self.push_inference.value==0 or self.show_results.value==1:
            self.loop_cnt = 0
            time.sleep(0.01)
            return False
        # キューが空ではない場合
        elif not self.q.empty():
            time.sleep(0.01)
            return False

        if self.loop_cnt==0:
            with open(self.set_class_path, 'r', encoding='utf-8') as yml:
                self.set_class_data = yaml.load(yml, Loader=yaml.SafeLoader)
            self.results = []
            self.judgement = None
            self.s_time = 0
        return True

    def make_save_dir(self):
        """
        検出日時で保存ディレクトリ作成
        検出時刻で保存名
        """
        date_y = datetime.datetime.strftime(datetime.datetime.now(), '%Y')
        date_m = datetime.datetime.strftime(datetime.datetime.now(), '%m')
        date_d = datetime.datetime.strftime(datetime.datetime.now(), '%d')
        self.date_name = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S%f')[:-4]
        self.save_img_dir = Path(self.project) / 'images' / date_y / date_m / date_d
        if self.save_img:
            self.save_img_dir.mkdir(parents=True, exist_ok=True)
        if self.save_txt:
            self.save_txt_dir = Path(self.project) / 'labels' / date_y / date_m / date_d
            self.save_txt_dir.mkdir(parents=True, exist_ok=True)
        if self.save_evidence:
            self.save_evidence_dir = Path(self.evidence_project) / date_y / date_m / date_d
            self.save_evidence_dir.mkdir(parents=True, exist_ok=True)
        if self.save_crop:
            self.save_crop_dir = Path(self.project) / 'crops' / date_y / date_m / date_d
            self.save_crop_dir.mkdir(parents=True, exist_ok=True)

    def drawing_box(self, gn, det, raw_results, raw_crops, raw_txt_data, annotator, imc):
        """
        boxの描画(テキストデータ、画像、クロップ様)
        """
        # Write results
        for *xyxy, conf, cls in reversed(det):
            if self.save_txt:  # Write to file
                txt_path = str(self.save_txt_dir / f'{self.date_name}')  # im.txt
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                raw_txt_data += (('%g ' * len(line)).rstrip() % line + '\n')

            # アノテーション
            if self.save_img or self.save_crop or self.save_evidence or self.save_mov:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True),
                                    class_flg=self.names[int(cls)] in self.set_class_data["target_names"])
                if self.save_crop:
                    # save_one_box(xyxy, imc, file=self.save_crop_dir / f'{self.date_name}_{str(c)}.jpg', BGR=True)
                    raw_crops.append([xyxy, imc, self.save_crop_dir / f'{self.date_name}_{str(c)}.jpg'])

            # 検知した対象
            if self.names[int(cls)] in self.set_class_data["target_names"]:
                raw_results.append(self.names[int(cls)])
        return raw_txt_data, raw_crops, raw_results

    def save(self, raw_txt_data, im0, raw_crops):
        """
        save data
        """
        # evidence ()
        if self.save_evidence:
            if self.judgement==True:
                cv2.imwrite(self.OK_evidence_path + '.jpg', self.OK_img)
            elif self.judgement==False:
                cv2.imwrite(self.NG_evidence_path + '.jpg', self.NG_img)
        # img
        if self.save_img:
            img_path = str(self.save_img_dir / f'{self.date_name}')  # im.txt
            cv2.imwrite(img_path + '.jpg', im0)
        # txt
        if self.save_txt and raw_txt_data:
            txt_path = str(self.save_txt_dir / f'{self.date_name}')  # im.txt
            with open(txt_path + '.txt', 'w') as f:
                f.write(raw_txt_data)
        # crop
        if self.save_crop:
            for crop in raw_crops:
                save_one_box(crop[0], crop[1], file=crop[2], BGR=True)

    @torch.no_grad()
    def run(self):
        """
        判定期間内(set_config_dataのjudgment_period)での検出結果をもとに
        検出対象(set_class_dataのtarget_names)が全て検出されたかを判定
        """
        self.loop_cnt = 0
        for path, im, im0s, vid_cap, s in self.dataset:
            if not self.observe_flg():    # フラグ監視と初期化
                continue

            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # save_dir
            self.make_save_dir()

            # Inference
            pred = self.model(im)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            # Process predictions
            for i, det in enumerate(pred):  # per image

                # if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), self.dataset.count

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for self.save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

                # Write results and get image with detections
                now_time = time.time()
                delta = now_time-self.s_time
                raw_results = []
                raw_crops = []
                raw_txt_data = ""
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    raw_txt_data, raw_crops, raw_results = self.drawing_box(gn, det, raw_results, raw_crops,
                                                                            raw_txt_data, annotator, imc)

                # 判定
                self.results.append(set(self.set_class_data["target_names"])==set(raw_results))

                # get image with detections
                im0, im0_evidence = annotator.result()
                if self.save_evidence and self.results[-1]:
                    self.OK_img = im0_evidence
                    self.OK_evidence_path = str(self.save_evidence_dir / f'{self.date_name}')
                elif self.save_evidence:
                    self.NG_img = im0_evidence
                    self.NG_evidence_path = str(self.save_evidence_dir / f'{self.date_name}')

                # 判定開始の処理、判定期間内での結果
                if delta >= self.set_config_data["judgment_period"]:
                    if len(self.results)==1 and\
                       self.set_config_data["trigger_name"] in {self.names[int(cls)] for *xyxy, conf, cls in reversed(det)} and\
                       self.judgement is None:
                        self.s_time = time.time()
                        self.judgement = "running"
                    elif self.judgement=="running":
                        self.judgement = True if True in self.results else False

                # Save results (image with detections)
                if self.judgement is not None:
                    self.save(raw_txt_data, im0, raw_crops)

                # 結果の共有
                self.q.put([imc, im0_evidence, self.judgement, self.results, raw_results])

                # 結果の初期化
                if self.judgement==True or self.judgement==False or self.judgement is None:
                    self.results = []
                    self.judgement = None

                # finish
                if self.close_flg.value==1:
                    return

                self.loop_cnt += 1
