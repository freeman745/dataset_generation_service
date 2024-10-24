from flask import Flask, request
import pickle
from modules.dataset.utils import get_order_files, generate_coco
from modules.image_compare import predict

from configs.config import ConfigWrapper
import traceback
import os
import threading
import queue
import cv2
from modules.inference import predict_orders


app = Flask(__name__)

backlog = {}
task_queue = queue.Queue()


def process_backlog(date, new_order, config):
    global backlog
    backlog[new_order] = 1

    for order_id in list(backlog.keys()):
        if backlog[order_id] > 5:
            backlog.pop(order_id, None)
            continue
        try:
            result = predict_orders(date, order_id, config)
            if result == 'success':
                backlog.pop(order_id, None)
            else:
                backlog[order_id] += 1
        except:
            backlog[order_id] += 1
            pass

    return 0


def task_worker():
    while True:
        date, order_id, config = task_queue.get()
        try:
            process_backlog(date, order_id, config)
        except Exception as e:
            print(f"Error processing order {order_id}: {e}")
        finally:
            task_queue.task_done()

worker_thread = threading.Thread(target=task_worker, daemon=True)
worker_thread.start()

config_path = 'configs/config.ini'
config = ConfigWrapper(config_path)
config.print_config()

pool = {'order_id': '', 'flag': 0}

def del_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)

# empty visible and ann folder
if os.path.isdir(config.image_path):
    del_files(config.image_path)
    print('Delete all the files in '+str(config.image_path))
if os.path.isdir(config.anns_path):
    del_files(config.anns_path)
    print('Delete all the files in '+str(config.anns_path))
# delete coco file
json_file_path = os.path.join(config.parent_folder, 'annotations_rgb.json')
if os.path.exists(json_file_path):
    os.remove(json_file_path)
    print('Delete '+str(json_file_path))

counting = 0

@app.route('/health', methods=['GET'])
async def health():
    res = {'msg':'ok', 'code':200}
    res = pickle.dumps(res)
    return res


@app.route('/generate', methods=['POST'])
async def generate():
    try:
        data = request.data
        data = pickle.loads(data)
        global pool
        if not pool['order_id']:
            res = {'msg':'this is the first request', 'code':200}
            res = pickle.dumps(res)
            pool['order_id'] = data['order_id']
            pool['flag'] = data['generate_flag']
            return res
        if pool['order_id']==data['order_id']:
            res = {'msg':"this order hasn't finished", 'code':200}
            res = pickle.dumps(res)
            return res

        date = data['date']
        order_id = pool['order_id']
        pool['order_id'] = data['order_id']
        flag = pool['flag']
        pool['flag'] = data['generate_flag']

        config.save = '1' if flag else '0'

        try:
            config.batch_size = int(data['batch_size'])
        except:
            pass

        try:
            config.padding = float(data['padding'])
        except:
            pass

        try:
            config.template_match = int(data['template_match'])
        except:
            pass

        task_queue.put((date, order_id, config))
        global backlog
        print(backlog)

        res = {'msg':"dataset generation start", 'code':200}
        res = pickle.dumps(res)
    
    except Exception as e:
        res = {'msg':str(e), 'code':300}
        res = pickle.dumps(res)
        traceback.print_exc()
    
    return res


@app.route('/compare', methods=['POST'])
async def compare():
    try:
        data = request.data
        data = pickle.loads(data)
        rgb0 = data['rgb0']
        rgb1 = data['rgb1']

        if len(rgb0.shape) == 3 and rgb0.shape[2] != 1:
            rgb0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2GRAY)
        if len(rgb1.shape) == 3 and rgb1.shape[2] != 1:
            rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)

        depth0 = data['depth0']
        depth0 = depth0[:, :, -1]
        depth1 = data['depth1']
        depth1 = depth1[:, :, -1]

        rgb_box, ori_points_rgb, ori_map_rgb, depth_box, ori_points_depth, ori_map_depth = predict.predict_one(rgb0, rgb1, depth0, depth1, config)

        res = {'msg':'success', 'code':200, 'mask':ori_map_depth}
        res = pickle.dumps(res)

    except Exception as e:
        res = {'msg':str(e), 'code':300, 'mask':''}
        res = pickle.dumps(res)
        traceback.print_exc()

    return res
    

if __name__ == '__main__':
    ip = config.ip
    port = config.port
    app.config['JSON_AS_ASCII'] = False
    app.run(host=ip, port=port, threaded=False)
