"""
File: concat_images.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: concat images
"""
from pathlib import Path

import cv2
import numpy as np

root_dir = '/home/nthere/2021/yolov5/utils/output/'
sub_dirs = ['nms_test', 'nms_test_after']

assert len(sub_dirs) > 0, 'no image sub directories specified'

filter_files = ['sample-2021-11-05.ece02b3a609c5234e4c0663f7b7060ff.bmp',
                'sample-2021-11-17.f7190a193ae0f70fac1f7b71fc3c9b8e.bmp',
                'sample-2021-11-18.7fff2af5f6bbb915879fb099e1389c05.bmp',
                'sample-2021-11-18.9e35f996081402d3111ee93fcba1b343.bmp',
                'sample-2022-01-05.a3dac461c0e116f619b2c4a7150fe784.bmp',
                'sample-2022-01-06.d366720887f0e65e38ba1b91e5875aac.bmp',
                'sample-2022-01-06.f20414e3e359688e695e7710b9fad69f.bmp',
                'sample-2022-01-24.704e506c9125d2a7b77d0a8d95455301.bmp',
                'sample-2022-01-27.42aa34755fffd7309e0fa06872328546.bmp',
                'sample-2022-01-27.480be31ff810e0eadb1fb7755bde6a7f.bmp',
                'sample-2022-01-27.4bd8e96adf714c4839e6ea8ccaa92c21.bmp',
                'sample-2022-01-27.71bbf88bda580c2a189e471af0fdcca1.bmp',
                'sample-2022-01-27.9759b2f00923e6216bd60df249b0d5e5.bmp',
                'sample-2022-01-27.986555d5628e051305d66659abd3aa5d.bmp',
                'sample-2022-01-27.a2ba16804226d96c1366b61244450e95.bmp',
                'sample-2022-01-27.b79aea3ffbb2022057537d4203d9e799.bmp',
                'sample-2022-01-27.c7df7f345f863a526175b1ca4a852f26.bmp',
                'sample-2022-01-27.de31617becd2c2afdfb1938c7edc8ac1.bmp',
                'sample-2022-01-27.f2376322a137fc78908d7c0bc482564e.bmp', ]

filter_files1 = [
    'sample-2021-11-05.46cc345d8c820e3a87b2df56dda89254.bmp',
    'sample-2021-11-18.28c009e254e679ce6982a7251712fb6b.bmp',
    'sample-2021-11-18.34740b094884f71b24d040aeccecfe2e.bmp',
    'sample-2021-11-18.90aa28880692282240fc9abd195b9a23.bmp',
    'sample-2021-11-18.90c1ee96d64eae080e2a9fa961358534.bmp',
    'sample-2021-11-18.9e35f996081402d3111ee93fcba1b343.bmp',
    'sample-2022-01-27.71bbf88bda580c2a189e471af0fdcca1.bmp']


filter_files = ['sample-2021-11-03.ca747cfe8ae9aca51a85c66d90d1bbb8.bmp',
                'sample-2021-11-17.1b5ffda21f0f18cb8ff832f8c6880f57.bmp',
                'sample-2021-11-18.7fff2af5f6bbb915879fb099e1389c05.bmp',
                'sample-2022-01-06.d366720887f0e65e38ba1b91e5875aac.bmp',
                'sample-2022-01-06.f20414e3e359688e695e7710b9fad69f.bmp',
                'sample-2022-01-24.f7ed23522484e678412b45406b8a1246.bmp',
                'sample-2022-01-27.42aa34755fffd7309e0fa06872328546.bmp',
                'sample-2022-01-27.480be31ff810e0eadb1fb7755bde6a7f.bmp',
                'sample-2022-01-27.71bbf88bda580c2a189e471af0fdcca1.bmp',
                'sample-2022-01-27.9759b2f00923e6216bd60df249b0d5e5.bmp',
                'sample-2022-01-27.986555d5628e051305d66659abd3aa5d.bmp',
                'sample-2022-01-27.a2ba16804226d96c1366b61244450e95.bmp',
                'sample-2022-01-27.b79aea3ffbb2022057537d4203d9e799.bmp',
                'sample-2022-01-27.c7df7f345f863a526175b1ca4a852f26.bmp',
                'sample-2022-01-27.dce583bd7bb78174b52e64474eaa806c.bmp',
                'sample-2022-01-27.de31617becd2c2afdfb1938c7edc8ac1.bmp',
                'sample-2022-01-27.f36c59749005caf4e42ad7df0dbc15af.bmp',
                'sample-2022-01-27.fde82ebc6b23e21de9bcac7624208c16.bmp']

filter_files = []

img_paths = []
# import pdb; pdb.set_trace()
for img_path in Path(root_dir).joinpath(sub_dirs[0]).rglob('*.png'):
    print(f'Processing: {img_path}')
    img_name = img_path.name
    img_paths.append(img_path)
    for sub_dir in sub_dirs[1:]:
        img_path = Path(root_dir) / sub_dir / img_name
        img_paths.append(img_path)

    imgs = []
    out_w, out_h = 480, 640
    for img_path in img_paths:
        imgs.append(cv2.resize(cv2.imread(str(img_path)), (out_w,
                                                           out_h)))
        curr_file = img_path.name.replace('debugWarped.', '')

    out_img = np.hstack(imgs)
    if curr_file in filter_files:
        cv2.imshow('output', out_img)
        cv2.waitKey(0)
    cv2.imwrite(f'./output_concat/{img_name}', out_img)
    img_paths = []
