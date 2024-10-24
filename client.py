import cv2
import requests
import pickle
import numpy as np
from tqdm import tqdm
import sys

import time


def test_dgs(url):
    order_list = ['4b3ca2e3460741fb943c35d17adc4a79','019dfab59b4b4de291e28f40fb1a1d67','69c0a770701f4dc19272713adb55d51d','793f7bbb66dd4cc0a7251fc7833f551c']
    order_list = ['007cec22b84d41d096ff4eec196eb671', '004929a4e94a449aa3e580713dfe46aa', '007cec22b84d41d096ff4eec196eb671']

    data = {
        'date' : '20240625',
        'order_id' : '',
        'generate_flag': 1
    }

    for i in tqdm(order_list):
        data['order_id'] = str(i)
        s_request = pickle.dumps(data)
        s_response = requests.post(url, data=s_request).content
        response = pickle.loads(s_response)

        print(response)


def test_one(url):
    from modules.image_compare.predict import read_compressed_file

    im0 = cv2.imread('test/data/db_data/kojiya_221229/Item/20240529/4b3ca2e3460741fb943c35d17adc4a79_3/PickDetectorDepth/detect/visible/2046346670.png')
    im1 = cv2.imread('test/data/db_data/kojiya_221229/Item/20240529/4b3ca2e3460741fb943c35d17adc4a79_3/PickDetectorDepth/extra/visible/2046447702.png')

    dp0 = read_compressed_file('test/data/db_data/kojiya_221229/Item/20240529/4b3ca2e3460741fb943c35d17adc4a79_3/PickDetectorDepth/detect/depth/2046347220.bin')
    dp1 = read_compressed_file('test/data/db_data/kojiya_221229/Item/20240529/4b3ca2e3460741fb943c35d17adc4a79_3/PickDetectorDepth/extra/depth/2046448087.bin')

    data = {
        'rgb0': im0,
        'rgb1': im1,
        'depth0': dp0,
        'depth1': dp1
    }

    s_request = pickle.dumps(data)
    s_response = requests.post(url, data=s_request).content
    response = pickle.loads(s_response)

    print(response['mask'])


if __name__ == "__main__":
    url = "http://127.0.0.1:4001/"
    s = time.time()
    if int(sys.argv[1]) == 0:
        test_dgs(url+'generate')
    if int(sys.argv[1]) == 1:
        test_one(url+'compare')
    print(time.time()-s)
