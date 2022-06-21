"""
show UI
"""

import tkinter as tk
from tkinter import ttk
import tkinter.font as font
import cv2
from PIL import Image, ImageTk
import time
import os
import sys
from pathlib import Path
import pygame
import datetime
import numpy as np
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



class Tk_app(tk.Frame):
    def __init__(self, master, q, set_config_path, class_path, set_class_path, sound_path,
                 push_inference, finish_load, close_flg, show_results):
        super().__init__(master)

        self.q = q

        self.set_config_path = set_config_path
        with open(self.set_config_path, 'r', encoding='utf-8') as yml:
            self.set_config_data = yaml.load(yml, Loader=yaml.SafeLoader)

        self.class_path = class_path

        self.set_class_path = set_class_path
        with open(self.set_class_path, 'r', encoding='utf-8') as yml:
            self.set_class_data = yaml.load(yml, Loader=yaml.SafeLoader)

        self.sound_path = sound_path
        self.push_inference = push_inference
        self.finish_load = finish_load
        self.close_flg = close_flg
        self.show_results = show_results

        # 親win
        # パラメータ
        self.app_bg='#003366'
        self.app_title="GenbaChecker"
        self.master.update_idletasks()
        self.app_w = self.master.winfo_screenwidth() # パソコン画面の幅を取得する関数
        self.app_h = self.master.winfo_screenheight() # パソコン画面の高さを取得する関数
        self.app_top = 0
        self.app_left = 0

        # 表示
        self.master.title(self.app_title)
        # self.master.geometry('1280x1920')
        geometry = "{:d}x{:d}+{:d}+{:d}".format(self.app_w, self.app_h, self.app_top, self.app_left)
        self.master.geometry(geometry)
        self.master.configure(bg=self.app_bg)
        # 親ウィンドウのグリッドを 1x1 にする
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.state("zoomed") # 最初から最大表示
        self.master.protocol("WM_DELETE_WINDOW", self.push_close_button)

        # 待機フレーム
        self.create_wait_frame()
        # 推論フレーム
        self.create_inference_frame()
        # 終了フレーム
        self.create_close_frame()

        # 画面更新の待機時間 ms
        self.update_delay = 100
        self.result_delay = 500

        # ロード完了後に設定画面を表示、推論画面のupdate
        self.wait(self.wait_frame, self.finish_load, 1, frame2=None, functions=[self.create_set_frame, self.update], shared=True)

    def create_wait_frame(self):
        """
        待機画面
        """
        self.wait_frame = tk.Frame(self.master, name="init")
        self.wait_frame.grid(row=0, column=0, sticky="nsew")
        # self.title("GENBA-Checker -起動画面-")
        # self.geometry("1000x1500")
        self.wait_frame.configure(bg='#003366')

        #text
        txt_title = tk.Label(self.wait_frame, text='GENBA-Checker', foreground='#ffffff', background='#003366', font=("Arial Black", 60))
        txt_progress = tk.Label(self.wait_frame, text='起動中', foreground='#ffffff', background='#003366', font=("Helvetica", 30))

        #image
        img_ixslogo = ImageTk.PhotoImage(file='./utils/img/ixslogo.png')
        #img_ixslogo = tk.PhotoImage(file='./utils/img/ixslogo.gif')
        ixslogo = ttk.Label(self.wait_frame, image=img_ixslogo)

        #progress
        style=ttk.Style()
        style.theme_use("clam")
        style.configure("blue.Horizontal.TProgressbar", troughcolor='#003366', background='#ffffff', bordercolor='#003366', darkcolor='#ffffff', lightcolor='#ffffff')
        progress = ttk.Progressbar(self.wait_frame, orient='horizontal', maximum=10, value=0, length=100, mode='indeterminate', style='blue.Horizontal.TProgressbar')
        progress.start()

        #place
        ixslogo.pack(pady=30)
        txt_title.pack(pady=30)
        txt_progress.pack(pady=30)
        progress.pack(pady=10)

    def create_set_frame(self):
        """
        セッティング画面
        """
        # モデルクラスのロード
        with open(self.class_path, 'r') as yml:
            self.class_data = yaml.load(yml, Loader=yaml.SafeLoader)

        self.set_frame = tk.Frame(self.master, name="set")
        self.set_frame.grid(row=0, column=0, sticky="nsew")
        # setting.title("GENBA-Checker -システム設定-")
        # setting.geometry("1000x1500")
        self.set_frame.configure(bg='#003366')

        #text
        txt_title = tk.Label(self.set_frame, text='GENBA-Checker', foreground='#ffffff', background='#003366', font=("Arial Black", 60))
        txt_1 = tk.Label(self.set_frame, text='アプリ設定', foreground='#ffffff', background='#003366', font=("Helvetica", 30))
        txt_2 = tk.Label(self.set_frame, text='検知項目を以下より選択してください', foreground='#ffffff', background='#003366', font=("Helvetica", 20))

        #place
        txt_title.pack(pady=5)
        txt_1.pack(pady=10)
        txt_2.pack(pady=10)

        #button
        name_convert = {n: [] for n in self.set_config_data["name_convert"].values()}
        for name, show_name in self.set_config_data["name_convert"].items():
            name_convert[show_name].append(name)
        name_convert = {"+".join(v): k for k, v in name_convert.items()}

        # name_convertを元に設定
        self.var_dic = {}
        for names, show_name in name_convert.items():
            tmp = []
            for name in names.split('+'):
                if name in self.class_data["all_names"] and not show_name: # 表示名なしの場合
                    tmp.append(None)
                elif name in self.class_data["all_names"]:
                    tmp.append(name in self.set_class_data["target_names"])
            # 複数nameある場合、tmp値が全てNoneかTureの場合が検出対象
            if set(tmp)=={None}:
                self.var_dic[names] = None
            elif set(tmp)=={True}:
                self.var_dic[names] = tk.BooleanVar()
                self.var_dic[names].set(True)
            elif tmp:
                self.var_dic[names] = tk.BooleanVar()
                self.var_dic[names].set(False)
            else:
                pass
            # ボタン設置
            if not set(tmp)=={None} and tmp:
                # 表示名ありでトリガーの場合は常にTrue
                if self.set_config_data["trigger_name"] in names.split('+'):
                    self.var_dic[names].set(True)
                    chk = tk.Checkbutton(self.set_frame, variable=self.var_dic[names], text=show_name,
                                         font=("Helvetica", 30), width=15, state=tk.DISABLED)
                else:
                    chk = tk.Checkbutton(self.set_frame, variable=self.var_dic[names], text=show_name, font=("Helvetica", 30), width=15)
                chk.pack(pady=10)

        start_button = tk.Button(self.set_frame, text="検出スタート", command=self.push_inference_button,
                                 font=("Helvetica", 30), relief="ridge", borderwidth="10", width=15)

        start_button.pack(pady=20)

    def create_inference_frame(self):
        """
        推論画面
        """
        # パラメータ
        judgment_period = self.set_config_data["judgment_period"]
        self.inference_top_label_text=f"カメラ前で{str(round(judgment_period))}秒間静止"
        self.inference_top_label_fg="white"
        self.inference_top_label_bg=self.app_bg
        self.inference_top_label_font=("Helvetica", "60")

        self.inference_date_label_text=datetime.datetime.strftime(datetime.datetime.now(), '%Y/%m/%d %H:%M')
        self.inference_date_label_fg="white"
        self.inference_date_label_bg=self.app_bg
        self.inference_date_label_font=("Helvetica", "30")

        self.inference_img_label_pad = 75 # タスクバーのサイズ
        self.inference_img_label_bg = self.app_bg

        self.inference_pause_button_text="アプリ終了"
        self.inference_pause_button_fg="black"
        self.inference_pause_button_bg="white"
        self.inference_pause_button_font=("Helvetica", "30")

        # フレーム
        self.inference_frame = tk.Frame(self.master, name="inference")
        self.inference_frame.grid(row=0, column=0, sticky="nsew")
        # self.inference_frame.title(self.app_title)
        # self.inference_frame.geometry(geometry)
        self.inference_frame.configure(bg=self.app_bg)
        # self.inference_frame.pack(fill = tk.BOTH)

        # ウィジェット作成
        # top text
        inference_top_label = tk.Label(self.inference_frame, text=self.inference_top_label_text,
                                  fg=self.inference_top_label_fg, bg=self.inference_top_label_bg,
                                  font=self.inference_top_label_font)
        # datetime
        inference_date_label = tk.Label(self.inference_frame, text=self.inference_date_label_text,
                                   fg=self.inference_date_label_fg, bg=self.inference_date_label_bg,
                                   font=self.inference_date_label_font)
        # exit Button
        inference_pause_button = tk.Button(self.inference_frame, text=self.inference_pause_button_text,
                                     fg=self.inference_pause_button_fg, bg=self.inference_pause_button_bg,
                                     font=self.inference_pause_button_font, relief="ridge", borderwidth="10")
        inference_pause_button.configure(command=self.push_pause_button)
        # img
        self.inference_get_img()
        inference_img_label = tk.Label(self.inference_frame, image=self.image_tk,
                                  width=self.image_pil.width, height=self.image_pil.height,
                                  bg=self.inference_img_label_bg,)

        # ウィジェットの配置
        inference_top_label.grid(column=0, columnspan=2, row=0,sticky=tk.NSEW)
        inference_date_label.grid(column=0, row=2, sticky=tk.NSEW)
        inference_pause_button.grid(column=1, row=2, sticky=tk.NSEW)
        inference_img_label.grid(column=0, columnspan=2, row=1, sticky=tk.NSEW)

        # 画面配列設定
        self.inference_frame.grid_columnconfigure(0, weight=1)
        self.inference_frame.grid_columnconfigure(1, weight=1)
        self.inference_frame.grid_rowconfigure(0, weight=1)
        self.inference_frame.grid_rowconfigure(2, weight=1)

    def create_close_frame(self):
        """
        終了画面
        """
        self.close_frame = tk.Frame(self.master, name="close")
        self.close_frame.grid(row=0, column=0, sticky="nsew")
        # setting.title("GENBA-Checker -システム設定-")
        # setting.geometry("1000x1500")
        self.close_frame.configure(bg='#333333')

        #text
        txt_title = tk.Label(self.close_frame, text='GENBA-Checker', fg='#ffffff', bg='#333333', font=("Arial Black", 60))
        txt_1 = tk.Label(self.close_frame, text='終了しますか？', fg='#ffffff', bg='#333333', font=("Helvetica", 60))

        #button
        start_button1 = tk.Button(self.close_frame, text="キャンセル",
                                  command=lambda: self.set_frame.tkraise(),
                                  font=("Helvetica", 30), relief="ridge", borderwidth="10", width=15)
        start_button2 = tk.Button(self.close_frame, text="終了", command=self.push_close_button,
                                 font=("Helvetica", 30, "bold"), relief="ridge", borderwidth="10", bg="#ED7D31", fg='red', width=15)

        #place
        txt_title.pack(pady=5)
        txt_1.pack(pady=10)
        start_button1.pack(pady=20)
        start_button2.pack(pady=20)

    def wait(self, frame1, flg, value, frame2=None, functions=None, shared=False):
        """
        flgが変化するまで待機する
        input
            frame1: 待機中に表示するフレーム
            flg: フラグ
            value: フラグが変化する値
            frame2: フラグ変化後に表示するフレーム
            functions: フラグ変化後の処理、リストで複数入力
            shared: multiprocessingのValueメモリの場合はTrue
        """
        frame1.tkraise()
        flg_value = flg.value if shared else flg
        if flg_value==value:
            if frame2:
                frame2.tkraise()
            for function in functions:
                frame2.after(10, function) if frame2 else frame1.after(10, function)
            return
        frame1.after(1000, self.wait, frame1, flg, value, frame2, functions, True)

    def push_inference_button(self):
        """
        推論ボタン
        """
        #スタートボタン押下後、メイン処理に引き数を渡す
        target_names = []
        go_inference = False
        # ボタン値の読み取り
        for names, var in self.var_dic.items():
            # varがNone(ボタンなし)は常に検出対象
            if var is None:
                target_names += names.split('+')
            elif var.get():
                go_inference = True
                target_names += names.split('+')
            else:
                pass
        # trigger_nameを加えset_configの更新
        self.set_class_data["target_names"] = list(set(target_names + [self.set_config_data["trigger_name"]]))
        with open(self.set_class_path, mode='w', encoding='utf-8') as yml:
            yaml.dump(self.set_class_data, yml, default_flow_style=False, sort_keys=False, encoding='utf-8', allow_unicode=True)

        # ボタンチェックがない場合は推論への以降不可
        if go_inference:
            self.push_inference.value = 1
            self.inference_frame.tkraise()
            go_inference = False
        else:
            pass

    def inference_get_img(self, img=None):
        """
        image_tk取得
        """
        if img is None:
            image_bgr = np.zeros((500, 500, 3)).astype('uint8')
        else:
            image_bgr = img
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
        w = image_pil.width # 横幅を取得
        h = image_pil.height # 縦幅を取得
        pad = 2 * (int(self.inference_top_label_font[1]) + max(int(self.inference_date_label_font[1]), int(self.inference_pause_button_font[1]))) # 表示文字の縦サイズ
        rate = (self.app_h-pad-self.inference_img_label_pad)/h # 表示文字の縦サイズ、タスクバーのサイズを考慮した画面に対する比率
        self.image_pil = image_pil.resize((int(w * rate), int(h * rate))) # リサイズ
        self.image_tk  = ImageTk.PhotoImage(self.image_pil) # ImageTkフォーマットへ変換

    def update(self):
        """
        推論時の画面更新
        """
        judgement = None
        # 時刻の更新
        self.inference_date_label_text=datetime.datetime.strftime(datetime.datetime.now(), '%Y/%m/%d %H:%M')
        self.inference_frame.grid_slaves(column=0, row=2)[0].config(text=self.inference_date_label_text)
        # 推論結果の取得
        if not self.q.empty():
            inference_results = self.q.get()
            raw_results = inference_results[4]
            judgement = inference_results[2]
            frame = inference_results[1]
            self.inference_get_img(frame)
            # OK,NG img
            if set(self.set_class_data["target_names"])==set(raw_results):
                self.OK_img = self.image_tk
            else:
                self.NG_img = self.image_tk
            # 結果表示
            if judgement is True:
                self.show_results.value=1
                self.inference_frame.grid_slaves(column=0, row=0)[0].config(text="チェックOK", bg="green")
                self.inference_frame.grid_slaves(column=0, row=1)[0].config(image=self.OK_img)
                # OK音
                if self.set_config_data["ok_sound"]:
                    pygame.mixer.Sound(self.sound_path/"OK_Buzzer.mp3").play()
            elif judgement is False:
                self.show_results.value=1
                self.inference_frame.grid_slaves(column=0, row=0)[0].config(text="チェックNG", bg="red")
                self.inference_frame.grid_slaves(column=0, row=1)[0].config(image=self.NG_img)
                # NG音
                if self.set_config_data["ng_sound"]:
                    pygame.mixer.Sound(self.sound_path/"NG_Buzzer.mp3").play()
            else:
                self.show_results.value=0
                judgment_period = self.set_config_data["judgment_period"]
                self.inference_frame.grid_slaves(column=0, row=0)[0].config(text=f"カメラ前で{str(round(judgment_period))}秒間静止",
                                                                            bg=self.inference_top_label_bg)
                self.inference_frame.grid_slaves(column=0, row=1)[0].config(image=self.image_tk)

        # repeat OK時は少し長めの待機
        if (judgement is True or judgement is False):
            self.show_results.value=1
            self.inference_frame.after(self.result_delay, self.update)
        else:
            self.show_results.value=0
            self.inference_frame.after(self.update_delay, self.update)

    def push_close_button(self):
        """
        終了ボタン
        """
        self.close_flg.value = 1
        self.master.destroy()

    def push_pause_button(self):
        """
        停止ボタン
        """
        self.push_inference.value = 0
        self.show_results.value = 0

        self.close_frame.tkraise()
