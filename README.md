# TrackCam
Transform QuickRoute file to video.

[Chinese](README_ZH.md)

## Install

```
git clone https://github.com/yaneeee/trackcam.git
cd trackcam
pip install -r requirements.txt
python main.py
```

## How to use
1. Analysis map picture and gpx data in QuickRoute.
2. Export QuickRoute data.
   - `File -> Export -> Route Data`
   - Data includes at least:
     - [x] time: str 
     - [x] imageX: float
     - [x] imageY: float 
     - [x] direction: float 
     - [x] speed: float 
     - [x] elapsedTimeFromStart: float 
     - [x] lapNumber: int
     - ...
3. Run `TrackCam`.
4. Choose map file, QuickRoute exported file and video file.
5. `Play` or 'Export'