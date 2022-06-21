"""
run inference and show UI
multiprocessing
"""

import time
import os
import sys
from pathlib import Path
from multiprocessing import Process, Queue, Value, Array
import yaml
import traceback

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def inference_thread(q, config_path, set_config_path, class_path, set_class_path, tmp_vars):
    from main.run_inference import Inference_app

    try:
        p = Inference_app(q, config_path, set_config_path, set_class_path, **tmp_vars)

        # モデル、データセットのロード
        p.load_model_data()

        # モデルのクラスを更新
        with open(class_path, 'r') as yml:
            class_data = yaml.load(yml, Loader=yaml.SafeLoader)
        class_data["all_names"] = p.names
        with open(class_path, mode='w') as yml:
            yaml.dump(class_data, yml, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # フラグ管理
        tmp_vars["finish_load"].value = 1
        while not tmp_vars["push_inference"].value:
            if tmp_vars["close_flg"].value==1:
                return
            time.sleep(1)

        # 推論
        p.run()

    except Exception:
        print(traceback.format_exc())
        sys.exit()

def tkapp_thread(q, set_config_path, class_path, set_class_path, sound_path, tmp_vars):
    import tkinter as tk
    import pygame
    from main.show_app import Tk_app

    try:
        root_tk = tk.Tk()

        # 音出すライブラリ初期化
        pygame.init()
        pygame.mixer.init()

        app = Tk_app(root_tk, q, set_config_path, class_path, set_class_path, sound_path, **tmp_vars) # Inherit
        app.mainloop()

    except Exception:
        print(traceback.format_exc())
        sys.exit()


class CheckerMain():
    def __init__(self,
                 config_path=ROOT / "main" / "config.yaml",
                 set_config_path=ROOT / "main" / "set_config.yaml",
                 class_path=ROOT / "main" / "class.yaml",
                 set_class_path=ROOT / "main" / "set_class.yaml",
                 sound_path=ROOT / "utils" / "Buzzer"):
        self.config_path = config_path
        self.set_config_path = set_config_path
        self.class_path = class_path
        self.set_class_path = set_class_path
        self.sound_path = sound_path
        pass
    def main(self):
        # 共有メモリ
        push_inference = Value('i', 0)    # 推論ボタン押打フラグ
        finish_load = Value('i', 0)    # モデルのロード完了フラグ
        close_flg = Value('i', 0)    # UI終了ボタン押打フラグ
        show_results = Value('i', 0)    # 結果表示中フラグ

        tmp_vars = vars().copy()
        del tmp_vars["self"]

        q = Queue()
        # UI
        p_tkapp = Process(target = tkapp_thread, args=(q, self.set_config_path, self.class_path,
                                                       self.set_class_path, self.sound_path, tmp_vars))
        # 推論
        p_inference = Process(target = inference_thread, args=(q, self.config_path, self.set_config_path,
                                                               self.class_path, self.set_class_path, tmp_vars))
        p_tkapp.start()
        p_inference.start()
        # どちらかが停止したら両方終了
        while p_tkapp.is_alive():
            if not p_inference.is_alive():
                p_tkapp.terminate()
                break
            time.sleep(1)
        p_tkapp.join()
        # p_tkapp終了したら推論も停止
        p_inference.terminate()
        p_inference.join()

def main():
    process = CheckerMain()
    process.main()

if __name__ == "__main__":
    main()
