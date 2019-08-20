#!/usr/bin/python

import json
from pathlib import Path
from PIL import Image as PILImage
#import IPython
import numpy as np
from math import trunc
import sys

# # CocoDataset Class
# This class imports and processes an annotations JSON file that you will specify when creating an instance of the class.
class CocoDataset():
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        
        # Customize these segmentation colors if you like, if there are more segmentations
        # than colors in an image, the remaining segmentations will default to white
        self.colors = ['red', 'green', 'blue', 'yellow']
        self.categories_of_interest = []
        self.person_count = 0
        self.car_count = 0
        self.bus_count = 0
        self.truck_count = 0
        self.vehicle_count = 0
        self.car_category_id = -1
        self.bus_category_id = -1
        self.truck_category_id = -1
        self.person_category_id = -1
        
        json_file = open(self.annotation_path)
        self.coco = json.load(json_file)
        json_file.close()
        
        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_segmentations()
        self._process_ourinterest_info()
    
    def _process_info(self):
        self.info = self.coco['info']
        
    def _process_licenses(self):
        self.licenses = self.coco['licenses']
        
    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()
        
        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']
            
            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')
            
            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {cat_id} # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}

            # Add code to record vehicle category IDs to be used later to count vehicle data
            if category["name"] == "car":
                self.categories_of_interest.append(cat_id)
                self.car_category_id = cat_id
            elif category["name"] == "bus":
                self.categories_of_interest.append(cat_id)
                self.bus_category_id = cat_id
            elif category["name"] == "truck":
                self.categories_of_interest.append(cat_id)
                self.truck_category_id = cat_id
            elif category["name"] == "person":
                self.categories_of_interest.append(cat_id)
                self.person_category_id = cat_id
            
    def _process_images(self):
        self.images = dict()
        for image in self.coco['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')
                
    def _process_segmentations(self):
        self.segmentations = dict()
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def _process_ourinterest_info(self):
        print('Process OurInterest Info')
        print('==================')
        self.person_count = 0
        self.car_count = 0
        self.bus_count = 0
        self.truck_count = 0
        self.vehicle_count = 0
        for key, items in self.segmentations.items():
            for item in items:
                if item["category_id"] in self.categories_of_interest:
                    if item["category_id"] == self.car_category_id:
                        self.car_count += 1
                        self.vehicle_count += 1
                    elif item["category_id"] == self.bus_category_id:
                        self.bus_count += 1
                        self.vehicle_count += 1
                    elif item["category_id"] == self.truck_category_id:
                        self.truck_count += 1
                        self.vehicle_count += 1
                    elif item["category_id"] == self.person_category_id:
                        self.person_count += 1
                        
        print('Category IDs of interest: ', self.categories_of_interest)
        print('Car category ID: ', self.car_category_id)
        print('Bus category ID: ', self.bus_category_id)
        print('Truck category ID: ', self.truck_category_id)
        print('Person category ID: ', self.person_category_id)

    def display_info(self):
        print('Dataset Info')
        print('==================')
        for key, item in self.info.items():
            print(f'  {key}: {item}')
            
    def display_licenses(self):
        print('Licenses')
        print('==================')
        for license in self.licenses:
            for key, item in license.items():
                print(f'  {key}: {item}')
                
    def display_categories(self):
        print('Categories')
        print('==================')
        for sc_name, set_of_cat_ids in self.super_categories.items():
            #print(f'  super_category: {sc_name}')
            for cat_id in set_of_cat_ids:
                if cat_id in self.categories_of_interest:
                    print(f'  super_category: {sc_name}')
                    print(f'    id {cat_id}: {self.categories[cat_id]["name"]}'
                    )
                    print('')
    
    def display_ourinterest_info(self):
        print('Our Interest Counts Info')
        print('==================')
        print('Category IDs of interest: ', self.categories_of_interest)
        print('Car category ID: ', self.car_category_id, ' #\(cars\): ', self.car_count)
        print('Bus category ID: ', self.bus_category_id, ' #\(buses\): ', self.bus_count)
        print('Truck category ID: ', self.truck_category_id, ' #\(trucks\): ', self.truck_count)
        print('Person category ID: ', self.person_category_id, ' #\(persons\): ', self.person_count)
        
    def display_image(self, image_id, show_bbox=True, show_polys=True, show_crowds=True):
        print('Image')
        print('==================')
        
        # Print image info
        image = self.images[image_id]
        for key, val in image.items():
            print(f'  {key}: {val}')
            
        # Open the image
        image_path = Path(self.image_dir) / image['file_name']
        image = PILImage.open(image_path)
        
        # Calculate the size and adjusted display size
        max_width = 600
        image_width, image_height = image.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height
        
        # Create bounding boxes and polygons
        bboxes = dict()
        polygons = dict()
        rle_regions = dict()
        seg_colors = dict()
        
        for i, seg in enumerate(self.segmentations[image_id]):
            if i < len(self.colors):
                seg_colors[seg['id']] = self.colors[i]
            else:
                seg_colors[seg['id']] = 'white'
                
            print(f'  {seg_colors[seg["id"]]}: {self.categories[seg["category_id"]]["name"]}')
            
            bboxes[seg['id']] = np.multiply(seg['bbox'], adjusted_ratio).astype(int)
            
            if seg['iscrowd'] == 0:
                polygons[seg['id']] = []
                for seg_points in seg['segmentation']:
                    seg_points = np.multiply(seg_points, adjusted_ratio).astype(int)
                    polygons[seg['id']].append(str(seg_points).lstrip('[').rstrip(']'))
            else:
                # Decode the RLE
                px = 0
                rle_list = []
                for j, counts in enumerate(seg['segmentation']['counts']):
                    if counts < 0:
                        print(f'ERROR: One of the counts was negative, treating as 0: {counts}')
                        counts = 0
                    
                    if j % 2 == 0:
                        # Empty pixels
                        px += counts
                    else:
                        # Create one or more vertical rectangles
                        x1 = trunc(px / image_height)
                        y1 = px % image_height
                        px += counts
                        x2 = trunc(px / image_height)
                        y2 = px % image_height
                        
                        if x2 == x1: # One vertical column
                            line = [x1, y1, 1, (y2 - y1)]
                            line = np.multiply(line, adjusted_ratio)
                            rle_list.append(line)
                        else: # Two or more columns
                            # Insert left-most line first
                            left_line = [x1, y1, 1, (image_height - y1)]
                            left_line = np.multiply(left_line, adjusted_ratio)
                            rle_list.append(left_line)
                            
                            # Insert middle lines (if needed)
                            lines_spanned = x2 - x1 + 1
                            if lines_spanned > 2: # Two columns won't have a middle
                                middle_lines = [(x1 + 1), 0, lines_spanned - 2, image_height]
                                middle_lines = np.multiply(middle_lines, adjusted_ratio)
                                rle_list.append(middle_lines)
                                
                            # Insert right-most line
                            right_line = [x2, 0, 1, y2]
                            right_line = np.multiply(right_line, adjusted_ratio)
                            rle_list.append(right_line)
                            
                if len(rle_list) > 0:
                    rle_regions[seg['id']] = rle_list
                                
                            
        
        # Draw the image
        html = '<div class="container" style="position:relative;">'
        html += f'<img src="{str(image_path)}" style="position:relative; top:0px; left:0px; width:{adjusted_width}px;">'
        html += '<div class="svgclass">'
        html += f'<svg width="{adjusted_width}" height="{adjusted_height}">'
        
        # Draw shapes on image
        if show_polys:
            for seg_id, points_list in polygons.items():
                for points in points_list:
                    html += f'<polygon points="{points}"                         style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]}; fill-opacity:0.5; stroke-width:1;" />'
        
        if show_crowds:
            for seg_id, line_list in rle_regions.items():
                for line in line_list:
                    html += f'<rect x="{line[0]}" y="{line[1]}" width="{line[2]}" height="{line[3]}"                         style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]};                         fill-opacity:0.5; stroke-opacity:0.5" />'
        
        if show_bbox:
            for seg_id, bbox in bboxes.items():
                html += f'<rect x="{bbox[0]}" y="{bbox[1]}" width="{bbox[2]}" height="{bbox[3]}"                     style="fill:{seg_colors[seg_id]}; stroke:{seg_colors[seg_id]}; fill-opacity:0" />'
        
        html += '</svg>'
        html += '</div>'
        html += '</div>'
        html += '<style>'
        html += '.svgclass {position: absolute; top:0px; left: 0px}'
        html += '</style>'
        
        return html
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine COCO JSON from one or more files into a single new file")

    parser.add_argument("-if", "--input_json_files",  type=str, dest="input_json_files",
                        help='Comma-separated COCO json files to combine')
    parser.add_argument("-ip", "--image_paths", type=str, dest="image_paths",
                        help='Comma-separated image paths to combine')
    parser.add_argument("-of", "--output_json_file", dest="output_json_file",
                        help="path to the combined output JSON file")

    args = parser.parse_args()
    print("Cmdline args: ", args)
    input_json_files = args.input_json_files.split(",")
    image_paths = args.image_paths.split(",")
    if len(input_json_files) != len(image_paths):
        print("= Something wrong. Please specify 1 image path for every input JSON file")
        sys.exit(-1)
        
    '''
    #instances_json_path = "/mnt/bigdrive1/cnn/yolact/data/coco/annotations/instances_val2017.json"
    #images_path = "/mnt/bigdrive1/cnn/yolact/data/coco/images"
    '''
    for idx,json_path in enumerate(input_json_files):
        coco_dataset = CocoDataset(json_path, image_paths[idx])
        coco_dataset.display_info()
        #coco_dataset.display_licenses()
        coco_dataset.display_categories()
        coco_dataset.display_ourinterest_info()
        

