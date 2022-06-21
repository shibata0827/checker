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
閾値などyolov5の設定　動画・推論・CPUのみのため最小限
'''
./main/config.yaml
'''

UI、検出対象の設定
'''
./main/set_config.yaml
'''

チェックボックスで指定したクラス
'''
./main/set_class.yaml
'''

モデル検出可能なクラス
'''
./main/class.yaml
'''

## Run
'''
python checker.py
'''

## 参照
[yolov5](https://github.com/ultralytics/yolov5)
