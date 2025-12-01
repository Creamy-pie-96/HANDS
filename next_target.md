# My design philosophy
**Gesture Detection Module for HANDS (Hand Assisted Navigation and Device System)**

**This module provides modular gesture detectors that work with MediaPipe hand landmarks.**
**All detectors operate on normalized coordinates (0..1) to be resolution-independent.**

**Design principles:**
- **Reuse existing utilities from math_utils.py (EWMA, euclidean, landmarks_to_array)**
- **Keep detectors stateless where possible; use small state objects for temporal logic**
- **Compute hand metrics once per frame and pass to all detectors**
- **Operate in normalized space; convert to pixels only for visualization**


# Previous extensions
**1.  Thumbs up. Logic will be simple: if only thumb is extended and thumb tip.y is less than pinky MCP then it's thumbs up**
**2.  Thumbs down. Logic will be simple: if only thumb is extended and thumb tip.y is greater than pinky MCP then it's thumbs down**
**3.  Thumbs up moving up. check if thumbs is up using the previous logic and if the velocity is more than a threshold in the up direction**
**4.  Thumbs up moving down. check if thumbs is up using the previous logic and if the velocity is more than a threshold in the down direction**
**5.  Thumbs down moving up. check if thumbs is down using the previous logic and if the velocity is more than a threshold in the up direction**
**6.  Thumbs down moving down. check if thumbs is down using the previous logic and if the velocity is more than a threshold in the down direction**

# Extentions:
**1. Have to improve the zoom logic**
**2. Have to improve the swipe logic its too sensitive maybe need to check for more times if hand movement is consistent and not sudden jarking**