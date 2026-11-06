"""
Dataset Generator for Spam Image Filtering using Class Histograms
DS307 Computer Vision - Project 12

This script generates synthetic spam and ham (legitimate) images reflecting
the realistic visual characteristics found in image spam (promotional banners,
flashy advertisements, discount vouchers) versus legitimate images
(landscapes, indoor scenes, natural objects, portraits, clean documents).
"""

import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

SPAM_COLORS = [
    [(255, 0, 50), (255, 255, 0)],      # Red & Yellow
    [(255, 102, 0), (255, 255, 255)],   # Bright Orange & White
    [(0, 200, 255), (255, 20, 147)],    # Cyan & Neon Pink
    [(255, 215, 0), (180, 0, 0)],       # Gold & Crimson
    [(50, 205, 50), (255, 255, 0)],     # Lime Green & Yellow
    [(220, 20, 60), (255, 255, 255)],   # Crimson & White
    [(138, 43, 226), (255, 215, 0)],    # Purple & Gold
]

SPAM_TEXTS = [
    ["WINNER!", "CLAIM $1000 NOW", "CALL 1-800-SPAM", "CLICK HERE!"],
    ["LIMITED OFFER!", "90% OFF SALE", "USE CODE: WIN90", "HURRY!"],
    ["CONGRATULATIONS!", "YOU WON A CAR", "CONFIRM DETAILS", "INSTANT REWARD"],
    ["HOT DEAL!", "BUY 1 GET 3 FREE", "SPECIAL PROMO", "ACT FAST"],
    ["CASINO BONUS!", "$500 FREE CHIPS", "NO DEPOSIT NEEDED", "PLAY NOW!"],
    ["EXCLUSIVE DISCOUNT", "VIP ACCESS ONLY", "EXPIRES TODAY", "TAP TO REDEEM"],
    ["URGENT ALERT!", "ACCOUNT PRIZE READY", "VERIFY IDENTITY", "RECEIVE CASH"],
    ["SECRET WEIGHT LOSS", "LOSE 20 LBS FAST", "MIRACLE PILL", "DOCTORS HATE THIS"]
]

HAM_PALETTES = [
    # Sky and landscape (blues, greens, earthy tones)
    {"sky": (135, 206, 235), "ground": (34, 139, 34), "accent": (139, 69, 19)},
    # Sunset landscape (soft orange, muted purple, dark ground)
    {"sky": (245, 160, 100), "ground": (50, 40, 60), "accent": (255, 200, 150)},
    # Ocean scene (deep blue, seafoam, sandy beach)
    {"sky": (176, 224, 230), "ground": (30, 80, 140), "accent": (238, 214, 175)},
    # Interior room / architecture (neutral beige, warm gray, wood brown)
    {"sky": (240, 235, 220), "ground": (160, 140, 120), "accent": (100, 75, 50)},
    # Forest scene (deep greens, moss, foliage)
    {"sky": (144, 238, 144), "ground": (20, 80, 20), "accent": (85, 107, 47)},
    # Clean document / notepad
    {"sky": (250, 250, 252), "ground": (240, 240, 245), "accent": (60, 60, 80)},
]


def create_spam_image(width=320, height=240, index=0):
    """Generate a synthetic spam banner with saturated colors, sharp text, and badges."""
    palette = random.choice(SPAM_COLORS)
    bg_color, fg_color = palette[0], palette[1]
    
    # Create image with background
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add flashy diagonal stripes or background geometric blocks
    pattern_type = index % 4
    if pattern_type == 0:
        # Diagonal warning stripes
        for x in range(-width, width * 2, 40):
            draw.polygon([(x, 0), (x + 20, 0), (x - 20, height), (x - 40, height)], fill=fg_color)
    elif pattern_type == 1:
        # Big starburst or radial badge
        cx, cy = width // 2, height // 2
        for angle in range(0, 360, 20):
            rad = np.radians(angle)
            rad2 = np.radians(angle + 10)
            p1 = (cx, cy)
            p2 = (int(cx + width * np.cos(rad)), int(cy + height * np.sin(rad)))
            p3 = (int(cx + width * np.cos(rad2)), int(cy + height * np.sin(rad2)))
            draw.polygon([p1, p2, p3], fill=fg_color)
    elif pattern_type == 2:
        # Split color blocks with border
        draw.rectangle([0, int(height * 0.45), width, height], fill=fg_color)
        draw.rectangle([10, 10, width - 10, height - 10], outline=(255, 255, 0), width=6)
    else:
        # High-contrast framed banner
        draw.rectangle([15, 15, width - 15, height - 15], fill=fg_color, outline=(0, 0, 0), width=5)

    # Draw badges / buttons
    btn_box = [int(width * 0.2), int(height * 0.7), int(width * 0.8), int(height * 0.9)]
    draw.rectangle(btn_box, fill=(255, 0, 0), outline=(255, 255, 0), width=3)
    
    # Text overlay
    text_set = random.choice(SPAM_TEXTS)
    # Header
    draw.text((25, 25), text_set[0], fill=(255, 255, 255))
    # Subtitle
    draw.text((25, 65), text_set[1], fill=(255, 255, 0))
    draw.text((25, 105), text_set[2], fill=(255, 255, 255))
    # Button text
    draw.text((int(width * 0.28), int(height * 0.74)), text_set[3], fill=(255, 255, 255))
    
    # Convert to OpenCV numpy array and add some artificial high-contrast noise
    np_img = np.array(img)
    # Add some high-contrast borders or text artifacts
    cv2.putText(np_img, text_set[0], (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(np_img, text_set[1], (25, 85), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(np_img, text_set[3], (int(width * 0.26), int(height * 0.82)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    return np_img


def create_ham_image(width=320, height=240, index=0):
    """Generate a synthetic legitimate image with organic gradients, smooth textures, landscapes, or clean notes."""
    palette = random.choice(HAM_PALETTES)
    sky_color = palette["sky"]
    ground_color = palette["ground"]
    accent_color = palette["accent"]
    
    scene_type = index % 5
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if scene_type == 0:
        # Landscape: Sky gradient + gentle hills + sun + trees
        for y in range(height):
            ratio = y / float(height)
            r = int(sky_color[0] * (1 - ratio * 0.5))
            g = int(sky_color[1] * (1 - ratio * 0.3))
            b = int(sky_color[2] * (1 - ratio * 0.2))
            img[y, :] = (r, g, b)
        
        # Sun
        cv2.circle(img, (int(width * 0.8), int(height * 0.25)), 25, (255, 240, 180), -1)
        
        # Mountains / Rolling hills
        points = []
        for x in range(0, width + 20, 20):
            hy = int(height * 0.55 + 25 * np.sin(x * 0.03) + 15 * np.cos(x * 0.05))
            points.append([x, hy])
        points.append([width, height])
        points.append([0, height])
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], ground_color)
        
        # Trees
        for tx in range(30, width - 20, 50):
            ty = int(height * 0.65 + 15 * np.sin(tx * 0.03))
            cv2.rectangle(img, (tx - 3, ty), (tx + 3, ty + 25), (60, 40, 20), -1)
            cv2.circle(img, (tx, ty), 12, (20, 100, 20), -1)
            
    elif scene_type == 1:
        # Smooth organic texture / foliage pattern
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        X, Y = np.meshgrid(x, y)
        pattern = (np.sin(X) * np.cos(Y) + 1.0) / 2.0
        for c in range(3):
            ch = ground_color[c] + pattern * (accent_color[c] - ground_color[c])
            img[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
            
    elif scene_type == 2:
        # Seashore / Water horizon
        # Sky
        img[:int(height * 0.45), :] = sky_color
        # Ocean
        for y in range(int(height * 0.45), int(height * 0.8)):
            grad = (y - height * 0.45) / (height * 0.35)
            r = int(np.clip(ground_color[0] * (1 - grad * 0.3), 0, 255))
            g = int(np.clip(ground_color[1] * (1 - grad * 0.2), 0, 255))
            b = int(np.clip(ground_color[2] * (1 + grad * 0.1), 0, 255))
            img[y, :] = [r, g, b]
        # Beach sand
        img[int(height * 0.8):, :] = accent_color
        # Smooth blur
        img = cv2.GaussianBlur(img, (7, 7), 0)
        
    elif scene_type == 3:
        # Interior / desk scene
        # Wall
        img[:int(height * 0.6), :] = sky_color
        # Desk surface
        img[int(height * 0.6):, :] = ground_color
        # Notebook / coffee cup
        cv2.rectangle(img, (int(width * 0.2), int(height * 0.68)), (int(width * 0.5), int(height * 0.92)), (245, 245, 240), -1)
        cv2.circle(img, (int(width * 0.75), int(height * 0.78)), 22, accent_color, -1)
        cv2.circle(img, (int(width * 0.75), int(height * 0.78)), 16, (80, 50, 30), -1)
        
    else:
        # Clean document / presentation slide with natural subdued tone
        img[:] = (248, 249, 250)
        # Header line
        cv2.rectangle(img, (20, 20), (width - 20, 25), (100, 120, 150), -1)
        # Document title
        cv2.putText(img, "Research Notes & Observations", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        # Paragraph lines
        for ly in range(65, height - 20, 14):
            lw = random.randint(int(width * 0.5), width - 40)
            cv2.line(img, (25, ly), (lw, ly), (180, 180, 190), 2)
            
    # Add natural subtle camera sensor noise
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img


def generate_all_datasets(n_train_spam=50, n_train_ham=50, n_test_spam=20, n_test_ham=20):
    """Generate complete training and test sets."""
    splits = {
        "train": {
            "spam": (os.path.join(DATASET_DIR, "train", "spam"), n_train_spam, create_spam_image),
            "ham": (os.path.join(DATASET_DIR, "train", "ham"), n_train_ham, create_ham_image)
        },
        "test": {
            "spam": (os.path.join(DATASET_DIR, "test", "spam"), n_test_spam, create_spam_image),
            "ham": (os.path.join(DATASET_DIR, "test", "ham"), n_test_ham, create_ham_image)
        }
    }
    
    total_generated = 0
    for split_name, categories in splits.items():
        for cat_name, (out_path, count, gen_func) in categories.items():
            os.makedirs(out_path, exist_ok=True)
            for i in range(count):
                img = gen_func(width=320, height=240, index=i + (100 if split_name == "test" else 0))
                # OpenCV uses BGR by default, but PIL/NumPy arrays created above are RGB
                # Convert RGB to BGR before saving with cv2.imwrite
                bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                fname = f"{cat_name}_{i+1:03d}.png"
                cv2.imwrite(os.path.join(out_path, fname), bgr_img)
                total_generated += 1
            print(f"Generated {count} images in {out_path}")
            
    print(f"Dataset generation complete! Total images created: {total_generated}")


if __name__ == "__main__":
    generate_all_datasets()
