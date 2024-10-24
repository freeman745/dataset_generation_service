from modules.dataset.utils import get_order_files, generate_coco
from modules.image_compare import predict
import pymongo
from datetime import datetime, timedelta
import os
from modules.slack.base import BaseBot
import threading
from modules.rps_memory import MemoryHandler

memory_handler = MemoryHandler()

def count_files_with_jan(directory, jan):
    files = os.listdir(directory)

    count = 0
    for file_name in files:
        if os.path.isfile(os.path.join(directory, file_name)) and jan in file_name:
            count += 1
            
    return count

def get_jan_from_order(order_id, mongo_url):
    db_client = pymongo.MongoClient(mongo_url)
    collection = db_client['rps_logs']['Item']

    query = {'order_id': order_id+'_0'}

    jan_code = str(collection.find_one(query)['PickDetectorNode']['msg_dict'][0]['jan'])

    return jan_code


def get_order_from_jan(jan_code, mongo_url):
    db_client = pymongo.MongoClient(mongo_url)
    collection = db_client['rps_logs']['Item']

    now = datetime.now()
    start_of_day = datetime(now.year, now.month, now.day)
    end_of_day = start_of_day + timedelta(days=1)

    query = {
        "PickDetectorNode.jan": jan_code,
        "metadata.timestamp": {
            "$gte": start_of_day,
            "$lt": end_of_day
        }
    }
    projection = {'order_id': 1, '_id': 0}
    results = list(collection.find(query, projection))

    order_list = list(set([i['order_id'].split('_')[0] for i in results]))

    return order_list


def insert2memory(order_id, jan_code, iou):
    memory_handler.insert_value(feature_name='ol_iou',
                                jan=jan_code,
                                value=iou,
                                order_id=order_id,
                                is_successful=False,
                                max_keep=100
                                )


def predict_orders(date, order_id, config):
    model_monitor = BaseBot(config.bot_token)
    
    try:
        jan_code = get_jan_from_order(order_id, config.mongodb)
    except Exception as e:
        print(str(e))
        print('order: '+order_id+' search jan code fail')
        return 'fail'

    if not jan_code:
        print('order: '+order_id+' no jan code found')
        return 'fail'
    
    '''
    try:
        order_list = get_order_from_jan(jan_code, config.mongodb)
    except Exception as e:
        print(str(e))
        print('order: '+order_id+' jan code: '+jan_code+' search order fail')
        return 0
    
    if not order_list:
        print('order: '+order_id+' jan code: '+jan_code+' no order found')
        return 0
    '''
    source = os.path.join(config.source, str(date))

    rgb_list, depth_list, json_list, mask_list = get_order_files(source, order_id)

    if len(rgb_list) < 2:
        print('order: '+str(order_id)+' before or after pick result not found')

        return 'fail'
    
    save_flag = count_files_with_jan(config.image_path, jan_code) >= config.jan_max

    iou_list = predict.run(date, order_id, rgb_list, depth_list, json_list, mask_list, config, save_flag)
    print(order_id ,iou_list)

    for index in range(len(iou_list)):
        insert2memory(order_id + '_' + str(index), jan_code, iou_list[index])

        if iou_list[index] < config.iou_thresh:
            text = f'Find poor OL performance case. IOU score {iou_list[index]:.2f}\n'
            text += config.mongodb_url + order_id + '_' + str(index)
            threading.Thread(target=model_monitor.send_message, args=(config.channel, text,)).start()

    if config.save != '0':
        if count_files_with_jan(config.image_path, jan_code) >= config.jan_max:
            out_string = jan_code + ' images reach ' + str(config.jan_max)
            print(out_string)
            return 'success'

        e_count = len(os.listdir(config.image_path))
        current_size = config.batch_size // 2 * config.iteration * (config.iteration + 1) + config.batch_size
        print(current_size)

        if e_count > current_size:
            coco_path = generate_coco(config.anns_path, config.parent_folder, config.labels, jan_code)
            print('coco generated')
            config.iteration += 1

    return 'success'
