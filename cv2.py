import imageio.v2 as imageio
import numpy as np
from scipy import ndimage
from PIL import Image

COLOR_BGR2RGB = 0
COLOR_BGR2GRAY = 1
FONT_HERSHEY_SIMPLEX = 0
CAP_PROP_FPS = 5
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FRAME_COUNT = 7
THRESH_BINARY = 0
RETR_EXTERNAL = 0
CHAIN_APPROX_SIMPLE = 0
IMREAD_COLOR = 1


def cvtColor(img, code):
    if code == COLOR_BGR2GRAY:
        return img.mean(axis=2).astype(np.uint8)
    elif code == COLOR_BGR2RGB:
        return img[:, :, ::-1]
    raise ValueError("Unsupported color conversion")


def absdiff(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def threshold(src, thresh, maxval, type_):
    mask = (src > thresh).astype(np.uint8) * maxval
    return thresh, mask


def findContours(mask, mode, method):
    labeled, _ = ndimage.label(mask)
    objects = ndimage.find_objects(labeled)
    contours = []
    for sl in objects:
        y1, y2 = sl[0].start, sl[0].stop
        x1, x2 = sl[1].start, sl[1].stop
        contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        contours.append(contour)
    return contours, None


def contourArea(cnt):
    x1, y1 = cnt[0]
    x2, y2 = cnt[2]
    return float((x2 - x1) * (y2 - y1))


def boundingRect(cnt):
    x_coords = cnt[:, 0]
    y_coords = cnt[:, 1]
    x1, x2 = x_coords.min(), x_coords.max()
    y1, y2 = y_coords.min(), y_coords.max()
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def resize(img, size):
    """Nearest-neighbour resize supporting (width, height)."""
    w, h = size
    pil = Image.fromarray(img)
    return np.array(pil.resize((w, h)))


def rectangle(img, pt1, pt2, color, thickness):
    x1, y1 = pt1
    x2, y2 = pt2
    img[y1:y2, x1:x2] = color
    return img


def putText(img, text, org, fontFace, fontScale, color, thickness):
    # no-op in stub
    return img


class VideoCapture:
    def __init__(self, path):
        try:
            self.reader = imageio.get_reader(path, format="ffmpeg")
        except Exception:
            class _Empty:
                def get_next_data(self):
                    raise StopIteration

                def get_meta_data(self):
                    return {"fps": 0, "size": (0, 0)}

                def close(self):
                    pass

            self.reader = _Empty()
        self.meta = self.reader.get_meta_data()

    def read(self):
        try:
            frame = self.reader.get_next_data()
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return True, frame.astype(np.uint8)
        except Exception:
            return False, None

    def get(self, prop):
        if prop == CAP_PROP_FPS:
            return self.meta.get('fps', 30)
        if prop == CAP_PROP_FRAME_WIDTH:
            return self.meta['size'][0]
        if prop == CAP_PROP_FRAME_HEIGHT:
            return self.meta['size'][1]
        if prop == CAP_PROP_FRAME_COUNT:
            return self.meta.get('nframes', 0)
        return 0

    def release(self):
        self.reader.close()


class VideoWriter:
    def __init__(self, path, fourcc, fps, size):
        self.writer = imageio.get_writer(path, fps=fps, format="ffmpeg")

    def write(self, frame):
        self.writer.append_data(frame)

    def release(self):
        self.writer.close()


def VideoWriter_fourcc(*args):
    return 0


def haveImageReader(_):
    return True


def haveImageWriter(_):
    return True
