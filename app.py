from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import piexif
import os
import random
import io
import zipfile
from PIL import Image
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ─── Pipeline functions ───────────────────────────────────────────────────────

def add_poisson_noise(image_array):
    img_float = image_array.astype(np.float64)
    shadow_mask = 1.0 - (img_float / 255.0)
    noise_strength = 2.5 + (shadow_mask * 4.0)
    noise = np.random.normal(0, 1, img_float.shape) * noise_strength
    noise[:,:,0] *= 1.12
    noise[:,:,2] *= 1.08
    return np.clip(img_float + noise, 0, 255).astype(np.uint8)

def lift_black_point(image_array):
    img_float = image_array.astype(np.float64)
    black_lift = np.array([11, 10, 9])
    dark_mask = img_float < 30
    for c in range(3):
        img_float[:,:,c] = np.where(
            dark_mask[:,:,c],
            np.maximum(img_float[:,:,c], black_lift[c]),
            img_float[:,:,c]
        )
    return np.clip(img_float, 0, 255).astype(np.uint8)

def apply_micro_geometry(image_array):
    h, w = image_array.shape[:2]
    angle = random.uniform(-0.8, 0.8)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    crop_px = random.randint(2, 6)
    top    = random.randint(0, crop_px)
    left   = random.randint(0, crop_px)
    bottom = h - random.randint(0, crop_px)
    right  = w - random.randint(0, crop_px)
    cropped = rotated[top:bottom, left:right]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

def smartphone_jpeg_compression(image_array):
    quality1 = random.randint(88, 94)
    _, buffer1 = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, quality1])
    decoded1  = cv2.imdecode(buffer1, cv2.IMREAD_COLOR)
    quality2  = random.randint(82, 88)
    _, buffer2 = cv2.imencode('.jpg', decoded1, [cv2.IMWRITE_JPEG_QUALITY, quality2])
    return buffer2, quality2

def generate_iphone_exif():
    days_ago   = random.randint(0, 30)
    photo_date = datetime.now() - timedelta(days=days_ago)
    photo_date = photo_date.replace(
        hour=random.randint(9, 21),
        minute=random.randint(0, 59),
        second=random.randint(0, 59)
    )
    date_str = photo_date.strftime('%Y:%m:%d %H:%M:%S').encode()

    iso     = random.choice([32, 40, 50, 64, 80, 100, 125, 160, 200])
    shutter = random.choice([(1,60),(1,80),(1,100),(1,120),(1,160),(1,200)])
    lat = 48.8566 + random.uniform(-0.05, 0.05)
    lon =  2.3522 + random.uniform(-0.05, 0.05)

    def to_dms(deg):
        d = int(deg)
        m = int((deg - d) * 60)
        s = int(((deg - d) * 60 - m) * 60 * 100)
        return [(d,1),(m,1),(s,100)]

    exif_dict = {
        '0th': {
            piexif.ImageIFD.Make:           b'Apple',
            piexif.ImageIFD.Model:          b'iPhone 15 Pro',
            piexif.ImageIFD.Software:       b'17.4.1',
            piexif.ImageIFD.DateTime:       date_str,
            piexif.ImageIFD.XResolution:    (72, 1),
            piexif.ImageIFD.YResolution:    (72, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
        },
        'Exif': {
            piexif.ExifIFD.DateTimeOriginal:      date_str,
            piexif.ExifIFD.DateTimeDigitized:     date_str,
            piexif.ExifIFD.ExposureTime:          shutter,
            piexif.ExifIFD.FNumber:               (178, 100),
            piexif.ExifIFD.ISOSpeedRatings:       iso,
            piexif.ExifIFD.FocalLength:           (686, 100),
            piexif.ExifIFD.FocalLengthIn35mmFilm: 26,
            piexif.ExifIFD.LensMake:              b'Apple',
            piexif.ExifIFD.LensModel:             b'iPhone 15 Pro back camera 6.86mm f/1.78',
            piexif.ExifIFD.ColorSpace:            1,
            piexif.ExifIFD.WhiteBalance:          0,
            piexif.ExifIFD.Flash:                 16,
            piexif.ExifIFD.ExposureMode:          0,
            piexif.ExifIFD.SceneCaptureType:      0,
        },
        'GPS': {
            piexif.GPSIFD.GPSLatitudeRef:  b'N',
            piexif.GPSIFD.GPSLatitude:     to_dms(lat),
            piexif.GPSIFD.GPSLongitudeRef: b'E',
            piexif.GPSIFD.GPSLongitude:    to_dms(lon),
            piexif.GPSIFD.GPSAltitudeRef:  0,
            piexif.GPSIFD.GPSAltitude:     (random.randint(30, 150), 1),
        },
        '1st': {}
    }
    return piexif.dump(exif_dict)

def process_image_bytes(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    img_rgb       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_processed = lift_black_point(img_rgb)
    img_processed = add_poisson_noise(img_processed)
    img_processed = apply_micro_geometry(img_processed)
    img_bgr       = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)

    jpeg_buffer, quality = smartphone_jpeg_compression(img_bgr)
    exif_bytes = generate_iphone_exif()

    pil_img = Image.open(io.BytesIO(jpeg_buffer.tobytes()))
    out_buf = io.BytesIO()
    pil_img.save(out_buf, format='JPEG', quality=quality, exif=exif_bytes)
    out_buf.seek(0)
    return out_buf

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/process', methods=['POST'])
def process():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'Aucune image reçue'}), 400

    if len(files) == 1:
        f           = files[0]
        result      = process_image_bytes(f.read())
        if result is None:
            return jsonify({'error': 'Impossible de traiter cette image'}), 400
        base_name   = os.path.splitext(f.filename)[0]
        return send_file(
            result,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'{base_name}_processed.jpg'
        )

    # Multiple files → ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            result = process_image_bytes(f.read())
            if result:
                base_name = os.path.splitext(f.filename)[0]
                zf.writestr(f'{base_name}_processed.jpg', result.read())

    zip_buf.seek(0)
    return send_file(
        zip_buf,
        mimetype='application/zip',
        as_attachment=True,
        download_name='images_traitees.zip'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
