#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import print_function

import cv2
import dlib
import fire
import json
import math
import numpy as np
import os
import skvideo.io
import skvideo.datasets
import sys

from skimage import io
from base64 import b64encode

stdout = getattr(sys.stdout, 'buffer', sys.stdout)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


def triangle_center(p1, p2, p3):
    x = (p1[0] + p2[0] + p3[0]) / 3
    y = (p1[1] + p2[1] + p3[1]) / 3
    return (x, y)


def triangle_area(p1, p2, p3):
    return ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))


def polygon_center(count, points):
    s = 0.0
    gx = 0.0
    gy = 0.0
    for i in range(2, count):
        s1 = triangle_area(points[0], points[i - 1], points[i])
        pt = triangle_center(points[0], points[i - 1], points[i])
        gx += s1 * pt[0]
        gy += s1 * pt[1]
        s += s1
    if s == 0.0:
        x = sum([p[0] for p in points]) / float(count)
        y = sum([p[1] for p in points]) / float(count)
        return (x, y)
    else:
        return (gx / s, gy / s)


def imgcat_for_iTerm2(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        if os.environ['TERM'].startswith('screen'):
            osc = b'\033Ptmux;\033\033]1337;File='
            st = b'\a\033\\\n'
        else:
            osc = b'\033]1337;File='
            st = b'\a\n'
        stdout.write(b'%ssize=%d;inline=1:%s%s' %
                     (osc, len(data), b64encode(data), st))


def detect_pix_format_of_video(filename):
    metadata = skvideo.io.ffprobe(filename)
    pix_fmt = metadata["video"]["@pix_fmt"]
    return pix_fmt


def metadata_of_video(filename):
    return skvideo.io.ffprobe(filename)["video"]


face_index = 0
frame_count = 0
yuv_fmt_list = ['yuv420p']


def collect_faces_from_video(filename, output, confidence, resize, prefix, zerofill, rotate, expansion, frame_skip, webcam, imgcat):
    global face_index
    global frame_count

    if frame_skip == 0:
        frame_skip = 1

    resize_flag = False
    sar = '1:1'
    pix_fmt = 'yuv420p'

    if webcam != '':
        print("video: built-in camera or webcam")
        cap = cv2.VideoCapture(webcam)
    else:
        print("video:", filename)
        metadata = metadata_of_video(filename)
        print("codec:",metadata["@codec_long_name"])
        sar = metadata["@sample_aspect_ratio"]
        print("SAR (sample_aspect_ratio):", sar)
        pix_fmt = detect_pix_format_of_video(filename)
        pix_fmt = metadata["@pix_fmt"]
        print("pix_fmt:", pix_fmt)
        print("duration:", metadata["@duration"])
        print("nb_frames:", metadata["@nb_frames"])

        # Calculate frame resize ratio
        # 1440x1080 and SAR4:3 -> 4/3=1.333.. x 1440 = 1920
        resize_rate_width, resize_rate_height = float(sar.split(':')[0]), float(sar.split(':')[1])
        if (resize_rate_width > resize_rate_height):
            base_resize_rate = resize_rate_height
        else:
            base_resize_rate = resize_rate_width
        resize_rate_width = resize_rate_width / base_resize_rate
        resize_rate_height = resize_rate_height / base_resize_rate

        # Check resize is necessary
        if (resize_rate_width != 1.0) or (resize_rate_height != 1.0):
            resize_flag = True

        cap = cv2.VideoCapture(filename)

    if (not cap.isOpened()):
        print("Unable to connect to camera or video")
        sys.exit()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        height, width = frame.shape[:2]

        if frame_count % frame_skip == 0:
            # print("frame:", frame_count)
            if resize_flag:
                # Resize according to SAR
                width = int(width * resize_rate_width)
                height = int(height * resize_rate_height)
                frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_CUBIC)

            # Check YUV or RGB
            if pix_fmt in yuv_fmt_list:
                dframe = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
            else:
                dframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            dets, scores, _idx = detector.run(dframe, 1)

            for (face, score) in zip(dets, scores):
                if (score > confidence):
                    collect_faces_from_frame(face, frame, width, height, output, confidence, resize, prefix, zerofill, rotate, expansion, webcam, imgcat)

        frame_count = frame_count + 1

    cap.release()



def collect_faces_from_image(filename, output, confidence, resize, prefix, zerofill, rotate, expansion, frame_skip, webcam, imgcat):
    global face_index

    print("image:", filename)
    img = io.imread(filename)
    try:
        # rgb
        if img.shape[2] == 3:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except IndexError:
        # gray
        frame = img
    height, width = frame.shape[:2]

    dets, scores, _idx = detector.run(img, 1)

    for (face, score) in zip(dets, scores):
        if (score > confidence):
            collect_faces_from_frame(face, frame, width, height, output, confidence, resize, prefix, zerofill, rotate, expansion, webcam, imgcat)


def collect_faces_from_frame(face_rect, frame, width, height, output, confidence, resize, prefix, zerofill, rotate, expansion, webcam, imgcat):
    left, top  = face_rect.left(), face_rect.top()
    right, bottom = face_rect.right(), face_rect.bottom()

    if not top < 0 and left < 0 and bottom > height and right > width:
        # ignore irregular rect
        return
    else:
        shape = predictor(frame, face_rect)
        left_eye_points = []
        right_eye_points = []
        for i in range(36, 42):
            left_eye_points.append((shape.part(i).x, shape.part(i).y))
        for i in range(42, 48):
            right_eye_points.append((shape.part(i).x, shape.part(i).y))
        left_eye = polygon_center(6, left_eye_points)
        right_eye = polygon_center(6, right_eye_points)

        eyes_center = ( (left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2 )
        pupil_distance = math.sqrt(
            (right_eye[1] - left_eye[1])**2 + (right_eye[0] - left_eye[0])**2)

        expansion_rate = expansion - 1.0
        pd_int = int(round(pupil_distance * expansion_rate))

        start_x = max(0, left - pd_int)
        start_y = max(0, top - pd_int)
        end_x = min(width, right + pd_int)
        end_y = min(height, bottom + pd_int)

        # ignore when not square
        w = end_x - start_x
        h = end_y - start_y
        if w != h:
            return

        if rotate:
            # rotate face
            radian = math.atan2(
                left_eye[1] - right_eye[1], right_eye[0] - left_eye[0])
            rot = cv2.getRotationMatrix2D(eyes_center, -np.rad2deg(radian), 1.0)
            im_affine = cv2.warpAffine(frame, rot, frame.shape[:2][::-1], flags=cv2.INTER_LANCZOS4)
            dst_img = im_affine[start_y:end_y, start_x:end_x]
        else:
            dst_img = frame[start_y:end_y, start_x:end_x]

        save_face(dst_img, output, resize, prefix, zerofill, imgcat)


def save_face(face_img, output, resize, prefix, zerofill, imgcat):
    global face_index

    if resize:
        resize = int(resize)
        face_img = cv2.resize(face_img, (resize, resize), interpolation = cv2.INTER_LANCZOS4)

    number_padded = str(face_index).zfill(zerofill)
    filename = prefix + number_padded + ".png"
    filepath = os.path.join(output, filename)

    if cv2.imwrite(filepath, face_img):
        print("Saved:", filepath)
    else:
        print("Error: Failed to save", filepath)
        sys.exit(1)

    face_index = face_index + 1

    if imgcat:
        imgcat_for_iTerm2(filepath)


def find_all_files(target_dir):
    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            yield os.path.join(root, filename)


img_file_ext = ['.png', '.PNG',
                '.jpg', '.JPG', '.jpeg', '.JPEG',
                '.gif', '.GIF']
video_file_ext = ['.mp4', '.MP4']


def collect(target='', output='faces', confidence=0.6, resize='', prefix='face', zerofill=5, rotate=False, expansion=1.5, frame_skip=30, webcam='', imgcat=False):
    # check target is specified
    if target == '':
        # check webcam option
        if webcam == '':
            print('Error: Please specify target.')
            sys.exit(1)
        else:
            collect_faces_from_video(target, output, confidence, resize, prefix, zerofill, rotate, expansion, frame_skip, webcam, imgcat)
    else:
        if webcam != '':
            print('`--target` option is ignored.')
            collect_faces_from_video(target, output, confidence, resize, prefix, zerofill, rotate, expansion, frame_skip, webcam, imgcat)

    if expansion < 1.0:
        print('Error: expansion value must be at least 1.0')
        sys.exit(1)

    # Check file or directory
    if (os.path.isdir(target)):
        for filename in find_all_files(target):
            collect(filename, output, confidence, resize, prefix, zerofill, rotate, expansion, frame_skip, webcam, imgcat)
    else:
        _filename, extension = os.path.splitext(target)
        if extension in img_file_ext:
            collect_faces_from_image(target, output, confidence, resize, prefix, zerofill, rotate, expansion, frame_skip, webcam, imgcat)
        elif extension in video_file_ext:
            collect_faces_from_video(target, output, confidence, resize, prefix, zerofill, rotate, expansion, frame_skip, webcam, imgcat)
        else:
            print("Skip:", target)


if __name__ == '__main__':
    fire.Fire(collect)
