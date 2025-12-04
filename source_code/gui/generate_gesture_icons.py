#!/usr/bin/env python3
"""
Generate gesture indicator icons for HANDS system.
Creates professional-looking gesture icons with appropriate colors and motion indicators.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

# Configuration
OUTPUT_DIR = Path(__file__).parent / 'stickers'
ICON_SIZE = 512  # High resolution for scaling
BG_COLOR = (255, 255, 255, 0)  # Transparent background
CIRCLE_BG = True  # Add circular background

# Color scheme for each gesture
GESTURE_COLORS = {
    'pointing': (52, 152, 219),       # Blue
    'pinch': (230, 126, 34),          # Orange
    'zoom': (155, 89, 182),           # Purple
    'swipe': (46, 204, 113),          # Green
    'open_hand': (241, 196, 15),      # Yellow/Gold
    'thumbs_up': (39, 174, 96),       # Bright Green
    'thumbs_down': (231, 76, 60),     # Red
    'thumbs_up_moving_up': (39, 174, 96),      # Green with blue arrow
    'thumbs_up_moving_down': (39, 174, 96),    # Green with red arrow
    'thumbs_down_moving_up': (231, 76, 60),    # Red with blue arrow
    'thumbs_down_moving_down': (231, 76, 60),  # Red with dark red arrow
}

# Emoji/Unicode characters for gestures
GESTURE_EMOJIS = {
    'pointing': '👆',
    'pinch': '🤏',
    'zoom': '🤌',
    'swipe': '👋',
    'open_hand': '✋',
    'thumbs_up': '👍',
    'thumbs_down': '👎',
    'thumbs_up_moving_up': '👍',
    'thumbs_up_moving_down': '👍',
    'thumbs_down_moving_up': '👎',
    'thumbs_down_moving_down': '👎',
}

# Arrow indicators for movement gestures
MOVEMENT_ARROWS = {
    'thumbs_up_moving_up': ('↑', (41, 128, 185)),      # Blue upward
    'thumbs_up_moving_down': ('↓', (192, 57, 43)),     # Red downward
    'thumbs_down_moving_up': ('↑', (41, 128, 185)),    # Blue upward
    'thumbs_down_moving_down': ('↓', (192, 57, 43)),   # Dark red downward
}

# Special decorations
GESTURE_DECORATIONS = {
    'zoom': [('⟷', (100, 100, 100))],  # Bidirectional arrows for zoom
    'swipe': [('⟷', (100, 100, 100))],  # Horizontal movement indicator
}


def create_gesture_icon(gesture_name, emoji, color, size=ICON_SIZE):
    """
    Create a gesture icon with emoji, optional arrow, and circular background.
    
    Args:
        gesture_name: Name of the gesture
        emoji: Unicode emoji character
        color: RGB tuple for the background color
        size: Icon size in pixels
        
    Returns:
        PIL Image object
    """
    # Create image with transparency
    img = Image.new('RGBA', (size, size), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Draw circular background
    if CIRCLE_BG:
        margin = size // 20
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=color + (255,),  # Add alpha channel
            outline=None
        )
    
    # Try to load emoji font (system-dependent)
    emoji_size = int(size * 0.5)
    try:
        # Common emoji font paths
        font_paths = [
            '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf',
            '/usr/share/fonts/truetype/ancient-scripts/Symbola_hint.ttf',
            '/System/Library/Fonts/Apple Color Emoji.ttc',
            'C:\\Windows\\Fonts\\seguiemj.ttf',  # Windows
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Fallback
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, emoji_size)
                    break
                except Exception:
                    continue
        
        if font is None:
            # Use default font as last resort
            font = ImageFont.load_default()
            
    except Exception as e:
        print(f"Warning: Could not load emoji font: {e}")
        font = ImageFont.load_default()
    
    # Calculate emoji position (centered)
    # For movement gestures, shift up to make room for arrow
    if gesture_name in MOVEMENT_ARROWS:
        emoji_y_offset = int(size * 0.3)
    else:
        emoji_y_offset = int(size * 0.5)
    
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), emoji, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    emoji_x = (size - text_width) // 2
    emoji_y = emoji_y_offset - text_height // 2
    
    # Draw emoji
    draw.text((emoji_x, emoji_y), emoji, font=font, fill=(255, 255, 255, 255), embedded_color=True)
    
    # Draw movement arrow if applicable
    if gesture_name in MOVEMENT_ARROWS:
        arrow_char, arrow_color = MOVEMENT_ARROWS[gesture_name]
        arrow_size = int(size * 0.3)
        
        try:
            arrow_font = ImageFont.truetype(font_paths[0] if os.path.exists(font_paths[0]) else None, arrow_size)
        except:
            arrow_font = ImageFont.load_default()
        
        # Position arrow
        if '↑' in arrow_char:
            arrow_y = int(size * 0.15)
        else:  # ↓
            arrow_y = int(size * 0.7)
        
        arrow_bbox = draw.textbbox((0, 0), arrow_char, font=arrow_font)
        arrow_width = arrow_bbox[2] - arrow_bbox[0]
        arrow_x = (size - arrow_width) // 2
        
        draw.text((arrow_x, arrow_y), arrow_char, font=arrow_font, fill=arrow_color + (255,))
    
    # Draw decoration symbols (like zoom indicators)
    if gesture_name in GESTURE_DECORATIONS:
        for decoration_char, decoration_color in GESTURE_DECORATIONS[gesture_name]:
            dec_size = int(size * 0.15)
            try:
                dec_font = ImageFont.truetype(font_paths[0] if os.path.exists(font_paths[0]) else None, dec_size)
            except:
                dec_font = ImageFont.load_default()
            
            # Position at bottom
            dec_bbox = draw.textbbox((0, 0), decoration_char, font=dec_font)
            dec_width = dec_bbox[2] - dec_bbox[0]
            dec_x = (size - dec_width) // 2
            dec_y = int(size * 0.75)
            
            draw.text((dec_x, dec_y), decoration_char, font=dec_font, fill=decoration_color + (255,))
    
    return img


def generate_all_icons():
    """Generate all gesture icons and save to output directory."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating gesture icons in: {OUTPUT_DIR}")
    print("-" * 60)
    
    generated_count = 0
    
    for gesture_name, emoji in GESTURE_EMOJIS.items():
        color = GESTURE_COLORS[gesture_name]
        
        # Generate icon
        icon = create_gesture_icon(gesture_name, emoji, color)
        
        # Save as PNG
        output_path = OUTPUT_DIR / f"{gesture_name}.png"
        icon.save(output_path, 'PNG')
        
        print(f"✓ Generated: {gesture_name}.png")
        generated_count += 1
    
    print("-" * 60)
    print(f"✓ Successfully generated {generated_count} gesture icons")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    # Also generate a combined preview image
    create_preview_grid()


def create_preview_grid():
    """Create a grid preview of all gestures for easy viewing."""
    gestures = list(GESTURE_EMOJIS.keys())
    n = len(gestures)
    
    # Calculate grid dimensions (try to make roughly square)
    cols = 4
    rows = (n + cols - 1) // cols
    
    icon_size = 256  # Smaller for preview
    margin = 20
    
    grid_width = cols * icon_size + (cols + 1) * margin
    grid_height = rows * icon_size + (rows + 1) * margin
    
    # Create preview image
    preview = Image.new('RGBA', (grid_width, grid_height), (240, 240, 240, 255))
    
    for idx, gesture_name in enumerate(gestures):
        row = idx // cols
        col = idx % cols
        
        x = margin + col * (icon_size + margin)
        y = margin + row * (icon_size + margin)
        
        # Load and resize icon
        icon_path = OUTPUT_DIR / f"{gesture_name}.png"
        if icon_path.exists():
            icon = Image.open(icon_path)
            icon = icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            preview.paste(icon, (x, y), icon)
    
    # Save preview
    preview_path = OUTPUT_DIR / '_preview_all_gestures.png'
    preview.save(preview_path, 'PNG')
    print(f"\n✓ Preview grid saved: {preview_path}")


if __name__ == '__main__':
    try:
        generate_all_icons()
        print("\n🎉 All gesture icons generated successfully!")
        print(f"\nTo use these icons, update your config.json:")
        print(f'  "stickers_base_path": "{OUTPUT_DIR}"')
    except Exception as e:
        print(f"\n❌ Error generating icons: {e}", file=sys.stderr)
        sys.exit(1)
