import cv2
import numpy as np
import pytest

from sweep.frames import extract_frames, resize_max, rotate_image, sharpness

from .conftest import closet_video_frames, write_hevc_mov, write_mp4


def test_sharpness_ranks_blur_lower():
    img = closet_video_frames(seconds=0.1)[0]
    assert sharpness(cv2.GaussianBlur(img, (15, 15), 5)) < 0.3 * sharpness(img)


def test_sampling_rate_and_keep_fraction(closet_video):
    frames = extract_frames(closet_video, fps=3, keep_fraction=1.0)
    assert 17 <= len(frames) <= 19                        # 6 s * 3 fps
    gaps = np.diff([f.time_s for f in frames])
    assert np.allclose(gaps, 1 / 3, atol=0.04)
    kept = extract_frames(closet_video, fps=3, keep_fraction=0.7)
    assert round(len(frames) * 0.7) <= len(kept) < len(frames)
    assert len(extract_frames(closet_video, fps=3, keep_fraction=0.7, min_relative=99)) == round(len(frames) * 0.7)
    assert [f.index for f in kept] == list(range(len(kept)))
    assert all(a.time_s < b.time_s for a, b in zip(kept, kept[1:]))


def test_blurry_frames_are_dropped(closet_video):
    kept = extract_frames(closet_video, fps=3, keep_fraction=0.7)
    assert not [f.time_s for f in kept if 2.0 <= f.time_s < 2.6], "motion-blurred frames should be dropped"


def test_resize_max(tmp_path):
    big = [np.full((1080, 1920, 3), 128, np.uint8) for _ in range(10)]
    big[0][::7] = 0
    frames = extract_frames(write_mp4(tmp_path / "big.mp4", big), fps=3, max_side=1024)
    assert frames[0].image.shape == (576, 1024, 3)
    small = np.zeros((300, 200, 3), np.uint8)
    assert resize_max(small, 1024) is small


def test_rotate_image():
    img = np.zeros((2, 3, 3), np.uint8)
    img[0, 0] = 255  # top-left marker
    r = rotate_image(img, 90)  # clockwise: top-left -> top-right
    assert r.shape == (3, 2, 3) and r[0, 1, 0] == 255
    assert rotate_image(img, 0) is img


@pytest.mark.parametrize("backend", ["opencv", "pyav"])
def test_iphone_hevc_portrait_rotation(tmp_path, backend):
    # Landscape-coded sensor frames, left half red; display matrix says
    # "rotate 90 deg clockwise" like a portrait iPhone clip.
    frames = []
    for _ in range(15):
        f = np.zeros((192, 320, 3), np.uint8)
        f[:, :160] = (255, 0, 0)
        frames.append(f)
    path = write_hevc_mov(tmp_path / "IMG_0001.MOV", frames, rotation_ccw=-90)
    out = extract_frames(path, fps=3, keep_fraction=1.0, backend=backend)
    img = out[0].image
    assert img.shape[:2] == (320, 192), "portrait after rotation"
    assert img[10, 96, 0] > 200 and img[10, 96, 2] < 60, "left half rotates to the top"
    assert img[-10, 96].max() < 40


def test_low_texture_stretch_is_not_dropped_wholesale():
    # First half: busy texture; second half: plain garment (sharp but flat);
    # one motion-blurred frame in each half. Only the blurred ones should go.
    from sweep.frames import relative_sharpness

    scores = np.array([900, 950, 300, 920, 910, 940, 60, 58, 20, 61, 59, 62], float)
    rel = relative_sharpness(scores, window=7)
    dropped = set(np.argsort(rel)[:2])
    assert dropped == {2, 8}
