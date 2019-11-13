from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import timeit

from flask import Flask
from flask import request, jsonify
from queue import Queue
from threading import Thread, Lock

# To pull files in URLs
from urllib.parse import urlparse
import requests

# To time responses
from timeit import default_timer as default_timeit_timer

app = Flask(__name__)
request_queue = None
worker_thread = None
contours_json_status_lock = None
contours_json = {}
start_queue_timer = None
end_queue_timer = None

@app.route("/api/v0/classify", methods = ['POST'])
@app.route("/api/v0/classify/", methods = ['POST'])
def handle_post():

    global start_queue_timer
    
    content = request.json
    #print(content)
    print(' * Queuing:', content)
    start_queue_timer = default_timeit_timer()
    request_queue.put(content)
    content["id"] = 0
    contours_json_status_lock.acquire()
    contours_json["status"] = "running"
    contours_json_status_lock.release()
    print(" * Set status to running")
    return jsonify(content), 201

@app.route("/api/v0/classify/<classify_id>", methods = ['GET'])
@app.route("/api/v0/classify/<classify_id>/", methods = ['GET'])
def handle_get(classify_id):
    # print(classify_id)
    if args.flask_debug_mode:
        print(" * Contours JSON in GET", json.dumps(contours_json))
    contours_json_status = None
    if contours_json:
        contours_json_status_lock.acquire()
        contours_json_status = contours_json["status"]
        contours_json_status_lock.release()
        
    if not contours_json:
        response_json = {"status": "idle"}
    elif (contours_json and contours_json_status == "running"):
        response_json = {"status": "running"}
    else:
        response_json = contours_json
    return jsonify(response_json)

def is_url(url):
      try:
              result = urlparse(url)
              return all([result.scheme, result.netloc])
      except ValueError:
              return False

def get_url_using_requests_lib(url_name, **kwargs):
        response = None
        retval_expect = 0
        
        try:
            if 'stream' in kwargs and kwargs['stream'] == True:
                response = requests.get(url_name, stream=True)
            elif 'headers' in kwargs and kwargs['headers']:
                response = requests.get(url_name, headers=kwargs['headers'])
            else:
                response = requests.get(url_name)
        except requests.URLRequired:
            print("Invalid URL: " + str(url_name))
            retval_expect = -1
        except requests.TooManyRedirects:
            print("Too many redirects for: " + str(url_name))
            retval_expect = -1
        except requests.ConnectionError:
            print("Connection error for URL: " + str(url_name))
            retval_expect = -1
        except requests.exceptions.RequestException:
            print("Ambiguous error for URL: " + str(url_name))
            retval_expect = -1
        else:
            if response.status_code == requests.codes.ok:
                retval_expect = 0
            else:
                retval_expect = -1
                
        return response, retval_expect
    
def flask_evaluate(thread_id, net:Yolact, dataset, input_queue):
    """This is the worker thread function.
    It processes items in the queue one after
    another.
    """
    global start_queue_timer, end_queue_timer
    
    print(" * Inside worker thread for flask_evaluate")

    net.detect.use_fast_nms = args.fast_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    while True:
        print(' * %s: Looking for the next item in queue' % thread_id)
        request_json = input_queue.get()
        end_queue_timer = default_timeit_timer()
        if start_queue_timer and end_queue_timer:
            print(" * Got item from queue in %0.4f ms: " % ((end_queue_timer - start_queue_timer) * 1000.0), request_json)
        else:
            print(" * Got item from queue: ", request_json)

        start_req_handling_timer = default_timeit_timer()
        
        # we just pretend and sleep
        if "output_filepath" in request_json:
            if request_json["output_filepath"].endswith(".jpg") or request_json["output_filepath"].endswith(".png"):
                out = request_json["output_filepath"]
            else:
                out = request_json["output_filepath"] + ".png"
        else:
            out = request_json["output_dir"] + request_json["input"].rsplit("/", 1)[1]
            # print(" * output_filepath: %s" % (out,))
            
        with torch.no_grad():
            evalimage(net, request_json["input"], out)
            
        input_queue.task_done()

        end_req_handling_timer = default_timeit_timer()
        print(" * Request handled in %0.4f ms" % ((end_req_handling_timer - start_req_handling_timer) * 1000),)
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--minsize', default=200, type=int,
                        help='Min. size to be used for image during inference/detection')
    parser.add_argument('--maxsize', default=700, type=int,
                        help='Max. size to be used for image during inference/detection')
    parser.add_argument('--contours_json', default=False, type=str2bool,
                        help='Should contours be generated for each object detected?')
    parser.add_argument('--contours_json_file', default=None, type=str,
                        help='A path to JSON file to be written out with detection info')
    parser.add_argument('--run_with_flask', default=False, type=str2bool,
                        help='Run with Flask and support a minimal REST-ful API?')
    parser.add_argument('--flask_port', default=11000, type=int,
                        help='Port to run web-service on')
    parser.add_argument('--flask_output_webserver', default="http://10.1.10.110:8080/openoutputs/", type=str,
                        help='Port to run web-service on')
    parser.add_argument('--flask_debug_mode', default=False, type=bool,
                        help='Accept HTTP requests and run in debug mode where its slower but with more info. printed out')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

    #print("= args.display = %d" % (int(args.display)))
    
iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        torch.cuda.synchronize()

    with timer.env('Copy'):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][:args.top_k]
            if args.contours_json:
                masks_cpu_numpy = masks.cpu().numpy().astype(np.uint8)
        classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]
    
    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    #for i in range(num_dets_to_consider):
    #    print(i, masks[i].nonzero().size())
    #print(masks[6].nonzero().size())
    if args.contours_json:
        contours_json["results"]["num_labels_detected"] = num_dets_to_consider
        
    if num_dets_to_consider == 0:
        if args.contours_json:
            if args.flask_debug_mode:
                print(" * About to set status to finished")
            if args.run_with_flask:
                contours_json_status_lock.acquire()
                contours_json["status"] = "finished"
                contours_json_status_lock.release()
                print(" * Set status to finished")

        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

        
    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
                    
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        #print(inv_alph_masks.size())
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    if args.display_text or args.display_bboxes or args.contours_json:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.contours_json:
                contours_json["results"]["left"].append(int(x1))
                contours_json["results"]["top"].append(int(y1))
                contours_json["results"]["right"].append(int(x2))
                contours_json["results"]["bottom"].append(int(y2))
                contours_json["results"]["confidence"].append(float(score))
                contours_json["results"]["labels"].append(cfg.dataset.class_names[classes[j]])
                
            if args.display_bboxes:
                #if (j == 6):
                #    print("x1,y1: %d,%d | x2,y2: %d,%d" % (x1,y1,x2,y2))
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
            
            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Code to get contours of the pixel masks
    if args.contours_json:
        mask_contours = []
        
        for j in range(num_dets_to_consider):
            #print("masks_cpu_numpy[%d] shape: " % j, masks_cpu_numpy[j].shape, " | dtype: ", masks_cpu_numpy[j].dtype)
            masks_cpu_numpy[j] *= 255

            # Find contours:
            contours, hierarchy = cv2.findContours(masks_cpu_numpy[j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.CHAIN_APPROX_NONE
            #print("%d contours len: " % j, len(contours), " hierarchy: ", hierarchy)
            if len(contours) == 0:
                pixel_mask_pts = list(masks[j].nonzero().size())[0]
                print(" * NO CONTOURS FOUND | # (PIXEL MASK PTS): ", pixel_mask_pts)
                print(" * %d contours len: " % j, len(contours), " hierarchy: ", hierarchy, " num non-zero pixel mask pts: ", pixel_mask_pts)
                mask_contours.append([])
            elif len(contours) == 1:
                contour_list = contours[0].tolist()
                mask_contours.append([contour_pt[0] for contour_pt in contour_list])
                #print("%d contours[0] len: " % j, len(contours[0]))
            else:
                #print("Contours 0:", contours[0])
                #print("Contours 1:", contours[1])
                max_len_contours_idx = 0
                max_len_contours = len(contours[0])
                for contour_idx in range(len(contours)):
                    if len(contours[contour_idx]) > max_len_contours:
                        max_len_contours = len(contours[contour_idx])
                        max_len_contours_idx = contour_idx

                contour_list = contours[max_len_contours_idx].tolist()
                mask_contours.append([contour_pt[0] for contour_pt in contour_list])
                #mask_contours.append(contours[max_len_contours_idx].tolist())
                #print("%d contours[%d] len: " % (j,max_len_contours_idx,), len(contours[max_len_contours_idx]))
            #if j == 6:
            #    print(mask_contours[j])

            contours_json["results"]["num_contour_points"].append(len(mask_contours[j]))
            contours_json["results"]["contours"].append(mask_contours[j])

        if args.flask_debug_mode:
            print(" * About to set status to finished")
        if args.run_with_flask:
            contours_json_status_lock.acquire()
            contours_json["status"] = "finished"
            contours_json_status_lock.release()
            print(" * Set status to finished")
                
    return img_numpy

def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k].cpu().numpy() for x in t]
    
    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()

def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]

def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x)*10)/10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })
    
    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)
    
    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                        'use_yolo_regressors', 'use_prediction_matching',
                        'train_masks']

        output = {
            'info' : {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)
        

        

def mask_iou(mask1, mask2, iscrowd=False):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """
    timer.start('Mask IoU')

    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    if iscrowd:
        # Make sure to brodcast to the right dimension
        ret = intersection / area1.t()
    else:
        ret = intersection / union
    timer.stop('Mask IoU')
    return ret.cpu()

def bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()

def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections:Detections=None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h*w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes  , gt_boxes   = split(gt_boxes)
                crowd_masks  , gt_masks   = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        scores = list(scores.cpu().numpy().astype(float))
        masks = masks.view(-1, h*w).cuda()
        boxes = boxes.cuda()


    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i,:],   scores[i])
                    detections.add_mask(image_id, classes[i], masks[i,:,:], scores[i])
            return
    
    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt   = len(gt_classes)

        mask_iou_cache = mask_iou(masks, gt_masks)
        bbox_iou_cache = bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item(), lambda i,j: crowd_bbox_iou_cache[i,j].item()),
            ('mask', lambda i,j: mask_iou_cache[i, j].item(), lambda i,j: crowd_mask_iou_cache[i,j].item())
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func in iou_types:
                gt_used = [False] * len(gt_classes)
                
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in range(num_pred):
                    if classes[i] != _class:
                        continue
                    
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(scores[i], True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue
                                
                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(scores[i], False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x =  ((x >> 16) ^ x) & 0xFFFFFFFF
    return x

def get_local_filepath(imgpath:str):

    local_filepath = None
    pulled_file = False

    if not is_url(imgpath):
        print("%s is not a url but a local file" % (imgpath))
        local_filepath = imgpath
    else:
        split_request_json = imgpath.rsplit("/", 1)
        #print(split_request_json)
        local_filepath = "/run/p1inputs/" + split_request_json[1]
        print(" * About to pull from %s into %s" % (imgpath, local_filepath,))
        response, retval = get_url_using_requests_lib(imgpath, stream=True)
        if response is not None:
            # print(" * http response status code: %d" % (response.status_code,))
            if (response.status_code == 200):
                try:
                    with open(local_filepath, 'wb') as fd:
                        fd.write(response.content)
                except IOError:
                    print("Could not write to file: " + str(local_filepath))
                else:
                    if args.flask_debug_mode:
                        print(" * Pulled from URL %s and wrote content to file %s of length %s" % (imgpath, local_filepath, response.headers.get('content-length')))
                    pulled_file = True
            else:
                print(" * Did not write content from %s to file %s as response status code is not 200/HTTP OK" % (imgpath, local_filepath,))
        else:
            print(" * Did not write content from %s to file %s as response is None" % (imgpath, local_filepath,))

    return local_filepath, pulled_file

def evalimage(net:Yolact, imgpath:str, save_path:str=None):    

    start_url_handling_timer = default_timeit_timer()
    path, url_valid_flag = get_local_filepath(imgpath)
    end_url_handling_timer = default_timeit_timer()
                
    local_img_cannot_be_read = False
    
    cv2_img_obj = None
    if url_valid_flag:
        # print(" * Path: %s" % (path,))

        start_file_read_timer = default_timeit_timer()
        cv2_img_obj = cv2.imread(path)
        #print(" * cv2_img_obj: %s" % (str(cv2_img_obj),))
        end_file_read_timer = default_timeit_timer()
        
        if cv2_img_obj is not None:    
            start_preds_timer = default_timeit_timer()
            try:
                frame = torch.from_numpy(cv2_img_obj).cuda().float()
            except TypeError:
                print(" * TypeError reading %s" % (path,))
                local_img_cannot_be_read = True
            else:
                batch = FastBaseTransform()(frame.unsqueeze(0))
                preds = net(batch)
            end_preds_timer = default_timeit_timer()
        else:
            print(" * OpenCV could not read input image file at: %s. Bad image file or invalid imgpath: %s?" % (path, imgpath,))
            local_img_cannot_be_read = True
    else:
        print(" * URL %s invalid" % (imgpath,))
        
    if args.contours_json:
        contours_json["input"] = imgpath
        contours_json["results"] = {}
        contours_json["results"]["left"] = []
        contours_json["results"]["top"] = []
        contours_json["results"]["right"] = []
        contours_json["results"]["bottom"] = []
        contours_json["results"]["confidence"] = []
        contours_json["results"]["labels"] = []
        contours_json["results"]["num_contour_points"] = []
        contours_json["results"]["contours"] = []

        if url_valid_flag == True and local_img_cannot_be_read == False:
            contours_json["output_urlpath"] = args.flask_output_webserver + save_path.rsplit("/", 1)[1]
            #contours_json["status"] = "running"

            start_prep_display_timer = default_timeit_timer()
            img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
            end_prep_display_timer = default_timeit_timer()
                        
            contours_json["error_description"] = ""
            
            #print(contours_json)
            if args.contours_json_file:
                with open(args.contours_json_file, "w") as ofp:
                    #json.dump(contours_json, ofp, sort_keys=True, indent=4, separators=(',', ': '))
                    json.dump(contours_json, ofp)
            else:
                if args.flask_debug_mode:
                    print(" * Contour JSON: ", json.dumps(contours_json))
                
            if save_path is None:
                img_numpy = img_numpy[:, :, (2, 1, 0)]

            if save_path is None:
                plt.imshow(img_numpy)
                plt.title(path)
                plt.show()
            else:
                start_outputimg_timer = default_timeit_timer()
                if args.flask_debug_mode:
                    cv2.imwrite(save_path, img_numpy)
                end_outputimg_timer = default_timeit_timer()

                url_handling_time_ms = ((end_url_handling_timer - start_url_handling_timer) * 1000)
                file_read_time_ms = ((end_file_read_timer - start_file_read_timer) * 1000.0)
                preds_time_ms = ((end_preds_timer - start_preds_timer) * 1000.0)
                prep_display_time_ms = ((end_prep_display_timer - start_prep_display_timer) * 1000.0)
                output_img_writeout_time_ms = ((end_outputimg_timer - start_outputimg_timer) * 1000.0)
                sum_time_ms = url_handling_time_ms + file_read_time_ms + preds_time_ms + prep_display_time_ms + output_img_writeout_time_ms
                print(" * URL-handling-time: %0.4f ms, File-read-time: %0.4f ms, Preds-time: %0.4f ms, Prep-display-timer: %0.4f ms, Output-writeout-timer: %0.4f ms, sum: %0.4f ms, FPS: %0.4f" % \
                      ( url_handling_time_ms, file_read_time_ms, preds_time_ms, prep_display_time_ms, output_img_writeout_time_ms, sum_time_ms, 1000.0/sum_time_ms))
            
        else:
            num_dets_to_consider = 0
            contours_json["results"]["num_labels_detected"] = num_dets_to_consider

            if args.flask_debug_mode:
                print(" * About to set status to finished")
            if args.run_with_flask:
                contours_json_status_lock.acquire()
                contours_json["status"] = "finished"
                contours_json_status_lock.release()
                print(" * Set status to finished")
                if local_img_cannot_be_read:
                    contours_json["error_description"] = "Invalid input file %s" % (imgpath,)
                elif not url_valid_flag:
                    contours_json["error_description"] = "Invalid input URL %s" % (imgpath,)
                    
def evalimages(net:Yolact, input_folder:str, output_folder:str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    for p in Path(input_folder).glob('*'): 
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        wrapped = wrapper(evalimage, net, path, out_path)
        time_taken = timeit.timeit(wrapped, number=1)
        #evalimage(net, path, out_path)
        print(path + ' -> ' + out_path + ' in time: %0.6f sec' % (time_taken,) + ' with an fps = %0.2f' % (1.0/time_taken,))
    print('Done.')

from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

def evalvideo(net:Yolact, path:str):
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()
    
    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:
        vid = cv2.VideoCapture(path)
    
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)
    
    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    # The 0.8 is to account for the overhead of time.sleep
    frame_time_target = 1 / vid.get(cv2.CAP_PROP_FPS)
    running = True

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        return [vid.read()[1] for _ in range(args.video_multiframe)]

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            return frames, net(imgs)

    def prep_frame(inp):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True)

    frame_buffer = Queue()
    video_fps = 0

    # All this timing code to make sure that 
    def play_video():
        nonlocal frame_buffer, running, video_fps, is_webcam

        video_frame_times = MovingAverage(100)
        frame_time_stabilizer = frame_time_target
        last_time = None
        stabilizer_step = 0.0005

        while running:
            frame_time_start = time.time()

            if not frame_buffer.empty():
                next_time = time.time()
                if last_time is not None:
                    video_frame_times.add(next_time - last_time)
                    video_fps = 1 / video_frame_times.get_avg()
                cv2.imshow(path, frame_buffer.get())
                last_time = next_time

            if cv2.waitKey(1) == 27: # Press Escape to close
                running = False

            buffer_size = frame_buffer.qsize()
            if buffer_size < args.video_multiframe:
                frame_time_stabilizer += stabilizer_step
            elif buffer_size > args.video_multiframe:
                frame_time_stabilizer -= stabilizer_step
                if frame_time_stabilizer < 0:
                    frame_time_stabilizer = 0

            new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)

            next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
            target_time = frame_time_start + next_frame_target - 0.001 # Let's just subtract a millisecond to be safe
            # This gives more accurate timing than if sleeping the whole amount at once
            while time.time() < target_time:
                time.sleep(0.001)


    extract_frame = lambda x, i: (x[0][i] if x[1][i] is None else x[0][i].to(x[1][i]['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)

    active_frames = []

    print()
    while vid.isOpened() and running:
        start_time = time.time()

        # Start loading the next frames from the disk
        next_frames = pool.apply_async(get_next_frame, args=(vid,))
        
        # For each frame in our active processing queue, dispatch a job
        # for that frame using the current function in the sequence
        for frame in active_frames:
            frame['value'] = pool.apply_async(sequence[frame['idx']], args=(frame['value'],))
        
        # For each frame whose job was the last in the sequence (i.e. for all final outputs)
        for frame in active_frames:
            if frame['idx'] == 0:
                frame_buffer.put(frame['value'].get())

        # Remove the finished frames from the processing queue
        active_frames = [x for x in active_frames if x['idx'] > 0]

        # Finish evaluating every frame in the processing queue and advanced their position in the sequence
        for frame in list(reversed(active_frames)):
            frame['value'] = frame['value'].get()
            frame['idx'] -= 1

            if frame['idx'] == 0:
                # Split this up into individual threads for prep_frame since it doesn't support batch size
                active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, args.video_multiframe)]
                frame['value'] = extract_frame(frame['value'], 0)

        
        # Finish loading in the next frames and add them to the processing queue
        active_frames.append({'value': next_frames.get(), 'idx': len(sequence)-1})
        
        # Compute FPS
        frame_times.add(time.time() - start_time)
        fps = args.video_multiframe / frame_times.get_avg()

        print('\rProcessing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d    ' % (fps, video_fps, frame_buffer.qsize()), end='')
    
    cleanup_and_exit()

def savevideo(net:Yolact, in_path:str, out_path:str):

    vid = cv2.VideoCapture(in_path)

    target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    four_cc = cv2.VideoWriter_fourcc(*"mp4v")
    print("\r FOURCC code: 0x%x" % (four_cc))
    out = cv2.VideoWriter(out_path, four_cc, target_fps, (frame_width, frame_height))
    #out = cv2.VideoWriter(out_path, 0x21, target_fps, (frame_width, frame_height), False)
    
    transform = FastBaseTransform()
    frame_times = MovingAverage()
    progress_bar = ProgressBar(30, num_frames)

    try:
        for i in range(num_frames):
            timer.reset()
            with timer.env('Video'):
                frame = torch.from_numpy(vid.read()[1]).cuda().float()
                batch = transform(frame.unsqueeze(0))
                preds = net(batch)
                processed = prep_display(preds, frame, None, None, undo_transform=False, class_color=True)

                out.write(processed)
            
            if i > 1:
                frame_times.add(timer.total_time())
                fps = 1 / frame_times.get_avg()
                progress = (i+1) / num_frames * 100
                progress_bar.set_val(i+1)

                print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                    % (repr(progress_bar), i+1, num_frames, progress, fps), end='')
    except KeyboardInterrupt:
        print('Stopping early.')
    
    vid.release()
    out.release()
    print()


def evaluate(net:Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out)
        else:
            evalimage(net, args.image)
        return
    elif args.images is not None:
        inp, out = args.images.split(':')
        evalimages(net, inp, out)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            savevideo(net, inp, out)
        else:
            evalvideo(net, args.video)
        return

    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()

    if not args.display and not args.benchmark:
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {
            'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }
        detections = Detections()
    else:
        timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))
    
    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        # Do a deterministic shuffle based on the image ids
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]

    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()

            with timer.env('Load Data'):
                img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

                # Test flag, do not upvote
                if cfg.mask_proto_debug:
                    with open('scripts/info.txt', 'w') as f:
                        f.write(str(dataset.ids[image_idx]))
                    np.save('scripts/gt.npy', gt_masks)

                batch = Variable(img.unsqueeze(0))
                if args.cuda:
                    batch = batch.cuda()

            with timer.env('Network Extra'):
                preds = net(batch)

            # Perform the meat of the operation here depending on our mode.
            if args.display:
                img_numpy = prep_display(preds, img, h, w)
            elif args.benchmark:
                prep_benchmark(preds, h, w)
            else:
                prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)
            
            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                frame_times.add(timer.total_time())
            
            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imshow(img_numpy)
                plt.title(str(dataset.ids[image_idx]))
                plt.show()
            elif not args.no_bar:
                if it > 1: fps = 1 / frame_times.get_avg()
                else: fps = 0
                progress = (it+1) / dataset_size * 100
                progress_bar.set_val(it+1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                    % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')



        if not args.display and not args.benchmark:
            print()
            if args.output_coco_json:
                print('Dumping detections...')
                if args.output_web_json:
                    detections.dump_web()
                else:
                    detections.dump()
            else:
                if not train_mode:
                    print('Saving data...')
                    with open(args.ap_data_file, 'wb') as f:
                        pickle.dump(ap_data, f)

                return calc_map(ap_data)
        elif args.benchmark:
            print()
            print()
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000*avg_seconds))

    except KeyboardInterrupt:
        print('Stopping...')


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    
    print_maps(all_maps)
    return all_maps

def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()



if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.minsize != 200 or args.maxsize != 700:
        print('Setting min. size to %d px and max. size to %d px' % (args.minsize, args.maxsize,))
        cfg.replace({'min_size': args.minsize, 'max_size': args.maxsize})
        print('cfg.min_size: %d, cfg.max_size: %d' % (cfg.min_size, cfg.max_size))
           
    print(' * display_bboxes: %d display_text: %d' % (int(args.display_bboxes), int(args.display_text)))
    
    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if args.image is None and args.video is None and args.images is None and args.run_with_flask is False:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None        

        print(' * Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        if args.run_with_flask:
            request_queue = Queue()
            contours_json_status_lock = Lock()
            worker_thread = Thread(target=flask_evaluate, args=(0, net, dataset, request_queue,))
            worker_thread.setDaemon(True)
            worker_thread.start()
            app.run(host='0.0.0.0', port=args.flask_port)
            #print '*** Main thread waiting'
            request_queue.join()
            #print '*** Done'
        else:
            evaluate(net, dataset)


