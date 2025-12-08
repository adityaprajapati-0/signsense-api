# app.py
# Flask backend for SignSense – receives MediaPipe landmarks and returns a sign label.

from flask import Flask, request, jsonify
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)  # allow browser calls from your PHP domain


# ------------- helpers -------------

def dist(a, b):
    """Euclidean distance in 2D between two landmarks."""
    if not a or not b:
        return 0.0
    ax, ay = float(a.get("x", 0.0)), float(a.get("y", 0.0))
    bx, by = float(b.get("x", 0.0)), float(b.get("y", 0.0))
    return math.hypot(ax - bx, ay - by)


def classify_sign(landmarks):
    """
    landmarks: list of dicts from JS:
      { x: 0-1, y: 0-1, z: ..., name: 'wrist' | 'thumb_tip' | ... }

    returns: (label: str, confidence: float, debug: dict)
    """

    if not isinstance(landmarks, list) or len(landmarks) < 21:
        return "Unknown", 0.2, {"error": "NOT_ENOUGH_LANDMARKS"}

    # Map by name for convenience
    by_name = {}
    for i, lm in enumerate(landmarks):
        name = lm.get("name")
        if name:
            by_name[name] = lm

    # Fallback access: by index if no name
    def g(name, idx):
        return by_name.get(name, landmarks[idx] if idx < len(landmarks) else None)

    # MediaPipe Hands standard order / names
    wrist = g("wrist", 0)

    thumb_tip  = g("thumb_tip", 4)
    thumb_ip   = g("thumb_ip", 3)

    index_tip  = g("index_finger_tip", 8)
    index_pip  = g("index_finger_pip", 6)

    middle_tip = g("middle_finger_tip", 12)
    middle_pip = g("middle_finger_pip", 10)

    ring_tip   = g("ring_finger_tip", 16)
    ring_pip   = g("ring_finger_pip", 14)

    pinky_tip  = g("pinky_finger_tip", 20)
    pinky_pip  = g("pinky_finger_pip", 18)

    if not wrist:
        return "Unknown", 0.1, {"error": "NO_WRIST"}

    # Finger extended check:
    # compare distance from wrist for tip vs pip
    def extended(tip, pip):
        if not tip or not pip:
            return False
        return dist(tip, wrist) > dist(pip, wrist) + 0.04  # 0.04 tuned for normalized coords

    thumb = extended(thumb_tip, thumb_ip)
    index = extended(index_tip, index_pip)
    middle = extended(middle_tip, middle_pip)
    ring = extended(ring_tip, ring_pip)
    pinky = extended(pinky_tip, pinky_pip)

    fingers = {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
    }

    count = sum(fingers.values())

    # For pinch detection, also consider thumb–index distance
    pinch_d = dist(thumb_tip, index_tip)

    # ------------- SIGN RULES (ORDER MATTERS) -------------

    # 🚽 Pinky only → TOILET
    if pinky and not thumb and not index and not middle and not ring:
        return "Pinky", 0.88, {**fingers, "reason": "TOILET"}

    # 😡 Middle only → ANGRY
    if middle and not thumb and not index and not ring and not pinky:
        return "Middle", 0.88, {**fingers, "reason": "ANGRY"}

    # 😎 Ring + Middle + Pinky → AWESOME
    if ring and middle and pinky and not thumb and not index:
        return "Ring Pinky Middle", 0.90, {**fingers, "reason": "AWESOME"}

    # 🤏 Pinch: thumb + index close together
    if thumb_tip and index_tip and pinch_d < 0.06:
        return "Pinch", 0.86, {**fingers, "pinch_distance": pinch_d}

    # ☝️ Point → index only
    if index and not middle and not ring and not pinky:
        return "Point", 0.87, {**fingers, "reason": "POINT"}

    # ✌️ Peace → index + middle
    if index and middle and not ring and not pinky:
        return "Peace", 0.88, {**fingers, "reason": "PEACE"}

    # 🤘 Rock → index + pinky
    if index and pinky and not middle and not ring:
        return "Rock", 0.88, {**fingers, "reason": "ROCK"}

    # 👍 Thumbs Up → thumb only
    if thumb and count == 1:
        return "Thumbs Up", 0.90, {**fingers, "reason": "THUMBS_UP"}

    # ✊ Fist → no fingers extended
    if count == 0:
        return "Fist", 0.85, {**fingers, "reason": "FIST"}

    # ✋ Open Hand → all 5 extended
    if count == 5:
        return "Open Hand", 0.92, {**fingers, "reason": "OPEN_HAND"}

    # 4 fingers
    if count == 4:
        return "Four Fingers", 0.82, {**fingers, "reason": "FOUR"}

    # 3 fingers
    if count == 3:
        return "Three Fingers", 0.80, {**fingers, "reason": "THREE"}

    return "Unknown", 0.30, {**fingers, "reason": "UNKNOWN"}


# ------------- ROUTES -------------

@app.route("/")
def home():
    return "SignSense Python API is running.\nPOST JSON with 'landmarks' to /api/handsign."


@app.route("/api/handsign", methods=["POST"])
def api_handsign():
    data = request.get_json(silent=True) or {}
    landmarks = data.get("landmarks")

    if not isinstance(landmarks, list) or not landmarks:
        return jsonify({
            "success": False,
            "error": "NO_LANDMARKS",
            "message": "No landmarks provided."
        }), 200

    sign, confidence, debug = classify_sign(landmarks)

    return jsonify({
        "success": True,
        "sign": sign,
        "confidence": confidence,
        "debug": debug
    }), 200


if __name__ == "__main__":
    # Local dev; on Render gunicorn will run app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
