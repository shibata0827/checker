# 内容
CPUで物体検知をリアルタイム検知、UI表示する  
・物体検知モデルyolov5を使用  
・検出対象はUIチェックボックス、set_config.yamlにて指定  
・指定時間内で目的のクラスが全て検出される画像があるかどうかで判定  
・検知結果をリアルタイムでUI表示するとともに、判定結果をOK,NGで表示  


## Install
```
git clone https://github.com/shibat](https://github.com/shibata0827/checker.git
cd checker
pip install -r requirements.txt
```

## Setting
閾値などyolov5の設定、動画・推論・CPUのみのため最小限
```
./main/config.yaml
```

UI、検出対象の設定
```
./main/set_config.yaml
```

## Run
```
python checker.py
```


## dir and file
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
