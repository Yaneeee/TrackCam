# TrackCam
将QuickRoute处理好的数据转换成视频

[English](README.md)

## 安装

```
git clone https://github.com/yaneeee/trackcam.git
cd trackcam
pip install -r requirements.txt
python main.py
```

## 使用
1. 在QuickRoute中配置地图和轨迹，以及时间分段
2. 导出Route Data数据
   - `File -> Export -> Route Data`
   - 导出数据时至少勾选以下项:
     - [x] time
     - [x] imageX
     - [x] imageY
     - [x] direction
     - [x] speed: float 
     - [x] elapsedTimeFromStart
     - [x] **lapNumber** `分段识别，必选`
     - ...
3. 运行TrackCam
4. 选择地图文件（JPEG), 轨迹数据文件（XML)，视频文件
5. 调整轨迹偏移、倍速播放、调整视图，播放预览
6. 导出视频