# Gesture Stickers

This directory contains sticker images for the HANDS status indicator.

## Required Sticker Files

Place PNG images (with transparency recommended) for each gesture:

| Gesture | Filename | Description |
|---------|----------|-------------|
| Pointing | `pointing.png` | Index finger pointing |
| Pinch | `pinch.png` | Thumb and index finger pinching |
| Zoom | `zoom.png` | Three-finger zoom gesture |
| Swipe | `swipe.png` | Four-finger swipe |
| Open Hand | `open_hand.png` | Five fingers spread |
| Thumbs Up | `thumbs_up.png` | Static thumbs up |
| Thumbs Down | `thumbs_down.png` | Static thumbs down |
| Thumbs Up Moving Up | `thumbs_up_moving_up.png` | Thumbs up moving upward |
| Thumbs Up Moving Down | `thumbs_up_moving_down.png` | Thumbs up moving downward |
| Thumbs Down Moving Up | `thumbs_down_moving_up.png` | Thumbs down moving upward |
| Thumbs Down Moving Down | `thumbs_down_moving_down.png` | Thumbs down moving downward |
| Pan | `pan.png` | Bimanual pan gesture |
| Precision Cursor | `precision_cursor.png` | Bimanual precision mode |

## Recommended Specifications

- **Format**: PNG with transparency
- **Size**: 64x64 or 128x128 pixels (will be scaled to fit indicator size)
- **Style**: Simple, clear icons with good contrast

## Custom Stickers

You can customize sticker paths in `config.json` under:
```json
"display": {
  "status_indicator": {
    "stickers_base_path": "stickers",
    "stickers": {
      "pointing": "pointing.png",
      ...
    }
  }
}
```

Set `stickers_base_path` to an absolute path or relative to the config file.
Individual sticker paths can also be absolute.

## Fallback Behavior

If a sticker file is not found or invalid, the status indicator will fall back to displaying a colored circle with an emoji:
- 🔴 Red circle: Paused / Error
- 🟡 Yellow circle: No hand detected
- 🔵 Blue circle: Hand detected with gesture emoji
