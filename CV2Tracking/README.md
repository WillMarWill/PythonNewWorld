# CV2Tracking

We implement the built-in trackers from OpenCV.

## Trackers
for one object, size of frame around 600*400
- medianflow (0.006x seconds per frame): very fast and our preferred tracker right now
- csrt(0.1x seconds per frame): very slow
- kcf(0.2) 
- boosting(error)
- mil(error)
- tld(error)
- mosse(0.1x s, not good accuracy)
## Usage

Clone this repo. The tracker is callable as follows: 
```python3
from CV2Tracking import Track

# initialize tracker
tracker = Track("kcf")

bboxes = tracker.track(old_bboxes,old_img,img)
```
