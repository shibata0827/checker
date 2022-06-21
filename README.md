# 内容
WEBCAMを使用しCPUで物体検知をリアルタイム検知、UI表示する  
・物体検知モデルyolov5を使用  
・検出対象はUIチェックボックス、set_config.yamlにて指定  
・評価時間内で目的のクラスが全て検出される画像があるかどうかで判定  
・検知結果をリアルタイムでUI表示するとともに、判定結果をOK,NGで表示  


## Install
```
git clone https://github.com/shibat](https://github.com/shibata0827/checker.git
cd checker
pip install -r requirements.txt
```

## Setting
閾値などyolov5の設定、動画・推論・CPUのみのため最小限  
以下はデフォルトyolov5から変更のあるパラメタ  
・weights: 対応形式onnx、pt  
・source: webcamのみ  
・save_txt: 評価時間内での検出結果を保存  
・save_conf: 評価時間内での検出結果を保存  
・save_crop: 評価時間内での検出結果を保存  
・save_img: 評価時間内での検出結果を保存  
・project: 指定ディレクトリ直下に日付毎のフォルダを作成し保存  
・save_evidence: 評価時間内での判定結果の根拠となる画像保存  
・camera_rot: カメラの向き [-90, 0, 90, 180]  
・evidence_project: 指定ディレクトリ直下に日付毎のフォルダを作成しsave_evidenceを保存  
```
./main/config.yaml
```

UI、検出対象の設定  
・name_convert: class nameとUI表示名の対応、class nameに存在する場合のみUI表示、表示名nullは常に検出対象  
  tie: null  
  clock: 持ち物  
  cell phone: 持ち物  
  cup: カップ  
・trigger_name: 検出開始のトリガー、常に検出対象  
・ok_sound: OK音の有無  
・ng_sound: NG音の有無  
・judgment_period: trigger_name検出から評価時間（秒）  
```
./main/set_config.yaml
```

## Run
```
python checker.py
```


## directory and file
推論実行ファイル
```
./main/run_inference.py
```

UI表示ファイル
```
./main/show_app.py
```

yolov5ファイル、動画・推論・CPU処理に必要なコードのみ
```
./main/models
./main/utils
```

yolov5学習済み重み、対応形式onnx・pt
```
./main/utils/weight
```

OK、NG判定時のブザー音
```
./main/utils/Buzzer
```

起動時のロード画像
```
./main/utils/img
```

チェックボックスで指定したクラス
```
./main/set_class.yaml
```

モデル検出クラス
```
./main/class.yaml
```


## 参照
[yolov5](https://github.com/ultralytics/yolov5)
