"""Build the labeled test set: data/test_set/images/*.jpg + data/test_set/labels.csv.

Sources:
  - data/source: frames from DocCornerDataset (MIDV-500/2019 specimen IDs, SmartDoc pages, COCO photos)
    with ground-truth corners in corners.json. Picked by eye, names in eval/selection.json.
  - data/own: own phone photos without personal data.
  - Synthetic defects made from the clean frames with fixed seeds. Geometric ones use the ground-truth
    corners, not the service's detector.

Usage: python eval/build_dataset.py
"""
import csv
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "data", "source")
OUT = os.path.join(ROOT, "data", "test_set")
OWN = os.path.join(ROOT, "data", "own")

rng = np.random.default_rng(7)


def load_sources() -> dict:
    corners = json.load(open(os.path.join(SOURCE, "corners.json"), encoding="utf-8"))
    return {name: (cv2.imread(os.path.join(SOURCE, "images", name)), None if quad is None else np.float32(quad))
            for name, quad in corners.items()}


def add_noise(im, sigma):
    return np.clip(im + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


def exposure(im, gain, offset, noise=0.0):
    out = np.clip(im.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    return add_noise(out, noise) if noise else out


def motion_blur(im, length):
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), float(rng.uniform(0, 180)), 1)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    return cv2.filter2D(im, -1, kernel / kernel.sum())


def jpeg(im, quality):
    return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)


def point_inside(quad):
    t = rng.uniform(0.3, 0.7, 2)
    top = quad[0] + (quad[1] - quad[0]) * t[0]
    bottom = quad[3] + (quad[2] - quad[3]) * t[0]
    return top + (bottom - top) * t[1]


def glare(im, quad):
    h, w = im.shape[:2]
    cx, cy = point_inside(quad)
    r = 0.12 * np.linalg.norm(quad[2] - quad[0])
    yy, xx = np.mgrid[:h, :w]
    spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))[..., None]
    return np.clip(im + 300 * spot, 0, 255).astype(np.uint8)


def cut_corner(im, quad):
    corner = int(rng.integers(4))
    side = 0.15 * max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
    x, y = quad[corner]
    h, w = im.shape[:2]
    left, top = corner in (0, 3), corner in (0, 1)
    x0, x1 = (int(x + side), w) if left else (0, int(x - side))
    y0, y1 = (int(y + side), h) if top else (0, int(y - side))
    return im[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]


# crop around the document first: perspective barely changes the shape of a small document,
# so tilting the full frame would only make it TOO_SMALL
def tilt(im, quad):
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
    h, w = im.shape[:2]
    im = im[int(max(y0 - my, 0)):int(min(y1 + my, h)), int(max(x0 - mx, 0)):int(min(x1 + mx, w))]
    h, w = im.shape[:2]
    k = 0.25
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[k * w, k * 0.6 * h], [(1 - k) * w, k * 0.6 * h], [w, h], [0, h]])
    border = tuple(int(v) for v in np.median(np.concatenate([im[0], im[-1]]), axis=0))
    return cv2.warpPerspective(im, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border)


# paint the document area as empty paper: base color + soft lighting gradient + sensor noise
def blank_document(im, quad):
    h, w = im.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    base = rng.uniform(205, 235)
    yy, xx = np.mgrid[:h, :w]
    paper = base + 12 * (xx / w - 0.5) + 8 * (yy / h - 0.5)
    paper = np.repeat(paper[..., None], 3, axis=2) + rng.normal(0, 2.5, (h, w, 3))
    out = im.copy()
    out[mask > 0] = np.clip(paper, 0, 255).astype(np.uint8)[mask > 0]
    return cv2.GaussianBlur(out, (3, 3), 0)


def screenshot(i):
    w, h = (1080, 2340) if i % 2 == 0 else (1170, 2532)
    dark = i % 3 == 0
    im = Image.new("RGB", (w, h), (30, 30, 30) if dark else (242, 242, 242))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 130], fill=(20, 20, 20) if dark else (255, 255, 255))
    d.text((40, 50), f"12:{40 + i}  LTE", fill=(200, 200, 200) if dark else (0, 0, 0))
    top = int(rng.integers(220, 420))
    d.rectangle([50, top, w - 50, top + int(rng.integers(1300, 1700))], fill=(255, 255, 255))
    for y in range(top + 70, top + 1250, int(rng.integers(32, 46))):
        d.rectangle([100, y, 100 + int(rng.integers(250, w - 220)), y + 14], fill=(40, 40, 40))
    d.rectangle([0, h - 170, w, h], fill=(20, 20, 20) if dark else (250, 250, 250))
    return jpeg(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), 85)


def main():
    selection = json.load(open(os.path.join(ROOT, "eval", "selection.json"), encoding="utf-8"))
    frames = load_sources()
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    rows = []

    def save(name, image, reasons, source, kind):
        cv2.imwrite(os.path.join(OUT, "images", name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"file": name, "ok": int(not reasons), "reasons": ";".join(reasons),
                     "source": source, "kind": kind})

    good = [(n, *frames[n]) for n in selection["good"]]
    own = {f: cv2.imread(os.path.join(OWN, f)) for f in sorted(os.listdir(OWN))}

    for name, im, _ in good:
        save(f"good_{name}", im, [], name, "real")
    for f in ("booklet.jpg", "leaflet_open_1.jpg", "leaflet_open_2.jpg", "folder.jpg"):
        save(f"good_own_{f}", own[f], [], f, "real")
    for name, im, _ in good[:8]:
        save(f"good_jpeg40_{name}", jpeg(im, 40), [], name, "synthetic")
    for name, im, _ in good[8:16]:
        save(f"good_dim_{name}", exposure(im, 0.6, 0, 3), [], name, "synthetic")

    for name in selection["negatives"]:
        save(f"nodoc_{name}", frames[name][0], ["NO_DOCUMENT"], name, "real")
    save("screenshot_own_pdf.jpg", own["screenshot_pdf.jpg"], ["SCREENSHOT"], "screenshot_pdf.jpg", "real")
    for i in range(9):
        save(f"screenshot_synthetic_{i}.jpg", screenshot(i), ["SCREENSHOT"], "generated", "synthetic")
    save("blank_own_sheet.jpg", own["blank_sheet.jpg"], ["BLANK_PAGE"], "blank_sheet.jpg", "real")
    for name, im, quad in good[16:23]:
        save(f"blank_{name}", blank_document(im, quad), ["BLANK_PAGE"], name, "synthetic")

    for name, im, quad in good[:15]:
        save(f"cut_{name}", cut_corner(im, quad), ["DOCUMENT_CUT"], name, "synthetic")
    for name in selection["small"]:
        save(f"small_{name}", frames[name][0], ["TOO_SMALL"], name, "real")
    for name, im, quad in good[15:25]:
        save(f"tilted_{name}", tilt(im, quad), ["TOO_TILTED"], name, "synthetic")

    for name in selection["real_blurry"]:
        save(f"blurry_{name}", frames[name][0], ["BLURRY"], name, "real")
    for name, im, _ in good[3:8]:
        save(f"blurry_gauss_{name}", add_noise(cv2.GaussianBlur(im, (0, 0), 3.0), 4), ["BLURRY"], name, "synthetic")
    for name, im, _ in good[20:25]:
        save(f"blurry_motion_{name}", add_noise(motion_blur(im, 20), 4), ["BLURRY"], name, "synthetic")

    for name, im, _ in good[5:13]:
        save(f"dark_{name}", exposure(im, 0.2, 0, 3), ["TOO_DARK"], name, "synthetic")
    for name, im, _ in good[13:21]:
        save(f"overexposed_{name}", exposure(im, 2.0, 50), ["OVEREXPOSED"], name, "synthetic")
    for name in selection["real_glare"]:
        save(f"glare_{name}", frames[name][0], ["GLARE"], name, "real")
    for name, im, quad in good[21:27]:
        save(f"glare_synthetic_{name}", glare(im, quad), ["GLARE"], name, "synthetic")

    with open(os.path.join(OUT, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} images ({sum(r['ok'] for r in rows)} good) -> {OUT}")


if __name__ == "__main__":
    main()
