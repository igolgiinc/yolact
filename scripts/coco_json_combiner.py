#!/usr/bin/python

import json
from pathlib import Path
from PIL import Image as PILImage
#import IPython
import numpy as np
from math import trunc
import sys
import coco_json_utils
from tqdm import tqdm

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
        self.our_interest_images = []
        self.our_interest_annotations = []
        
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
        self.coco_categories = []
        self.coco_category_ids_by_name = dict()
        
        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']
            cat_name = category['name']
            
            self.coco_categories.append({'supercategory': super_category,
                                         'id': cat_id,
                                         'name': cat_name})

            if cat_name not in self.coco_category_ids_by_name:
                self.coco_category_ids_by_name[cat_name] = cat_id
                
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

        print("Total categories (original): ", len(self.coco['categories']))
        
    def _process_images(self):
        self.images = dict()
        for image in self.coco['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')
        print("Total images (original): ", len(self.coco['images']))

    def _process_segmentations(self):
        self.segmentations = dict()
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)
        print("Total annotations (original): ", len(self.coco['annotations']), " | #images_with_annotations: ", len(self.segmentations.keys()))
        
    def _process_ourinterest_info(self):
        print('Process OurInterest Info')
        print('==================')
        self.person_count = 0
        self.car_count = 0
        self.bus_count = 0
        self.truck_count = 0
        self.vehicle_count = 0
        self.our_interest_images = []
        self.our_interest_annotations = []
        self.our_interest_images_set = set()
        # self.our_interest_annotations_set = set()
        
        #custom_super_categories = {"person": {'person'},
        #"vehicle": {'car', 'bus', 'truck'}}

        test_val = True
        for key, items in tqdm(self.segmentations.items()):
            for item in items:
                if item["category_id"] in self.categories_of_interest:
                    
                    image_id = key
                    #if self.images[image_id] not in self.our_interest_images:
                    #    self.our_interest_images.append(self.images[image_id])
                    if test_val:
                        test_val = False
                        print(isinstance(image_id, int), isinstance(image_id, str))

                    #if image_id > 1002100:
                    #    print(self.images[image_id])
                        
                    self.our_interest_images_set.add(image_id)
                    annotation_obj = item

                    if item["category_id"] == self.person_category_id:
                        self.person_count += 1
                        #annotation_obj["category_id"] = 1
                    elif item["category_id"] == self.car_category_id:
                        self.car_count += 1
                        self.vehicle_count += 1
                        #annotation_obj["category_id"] = 2
                    elif item["category_id"] == self.bus_category_id:
                        self.bus_count += 1
                        self.vehicle_count += 1
                        #annotation_obj["category_id"] = 3
                    elif item["category_id"] == self.truck_category_id:
                        self.truck_count += 1
                        self.vehicle_count += 1
                        #annotation_obj["category_id"] = 4

                    #if annotation_obj not in self.our_interest_annotations:
                    #    self.our_interest_annotations.append(annotation_obj)
                    # self.our_interest_annotations_set.add(annotation_obj)

        print("Appending from sets to lists")
        for image_id in tqdm(list(self.our_interest_images_set)):
            self.our_interest_images.append(self.images[image_id])
            for item in self.segmentations[image_id]:
                self.our_interest_annotations.append(item)
        #print(self.our_interest_images)
        print("Done converting sets to lists")
                
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
                #if cat_id in self.categories_of_interest:
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
        print('Num images of interest: ', len(self.our_interest_images), ' num annotations of interest: ', len(self.our_interest_annotations), " total_counts: ",
              self.car_count + self.bus_count + self.truck_count + self.person_count)
        
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

    parser = argparse.ArgumentParser(description="Generate COCO JSON for reduced set of categories")

    parser.add_argument("-if", "--input_json_files",  type=str, dest="input_json_files",
                        help='Comma-separated COCO json files to combine. The first one is the base')
    parser.add_argument("-ip", "--image_paths", type=str, dest="image_paths",
                        help='Comma-separated image paths to display image if needed (jupyter notebook use)')
    parser.add_argument("-of", "--output_json_file", dest="output_json_file",
                        help="Path to the combined output JSON file")
    
    args = parser.parse_args()
    #print("Cmdline args: ", args)

    input_json_files = args.input_json_files.split(",")
    image_paths = args.image_paths.split(",")
    
    cjc = coco_json_utils.CocoJsonCreator(skip_cmdline_args=True)
    
    coco_dataset_base = CocoDataset(input_json_files[0], image_paths[0])
    coco_dataset_to_combine = CocoDataset(input_json_files[1], image_paths[1])

    print("= Starting to create combined JSON")
    print("= Putting info, licenses from base")
    cjc.put_info(coco_dataset_base.info)
    cjc.put_licenses(coco_dataset_base.licenses)
    #print(coco_dataset.super_categories)
    #cjc.put_super_categories(custom_super_categories)
    print("= Putting categories info from base")
    cjc.put_categories_info(coco_dataset_base.coco_categories, coco_dataset_base.coco_category_ids_by_name)

    print("= Combining images and annotations info")
    combined_images_of_interest = coco_dataset_base.our_interest_images + coco_dataset_to_combine.our_interest_images
    combined_annotations_of_interest = coco_dataset_base.our_interest_annotations + coco_dataset_to_combine.our_interest_annotations
    
    cjc.put_images_info(combined_images_of_interest)
    cjc.put_annotations_info(combined_annotations_of_interest)
    
    #coco_dataset.display_info()
    #coco_dataset.display_licenses()
    coco_dataset_base.display_categories()
    coco_dataset_base.display_ourinterest_info()
    coco_dataset_to_combine.display_ourinterest_info()

    print("= About to enter cjc.main")
    cjc.main(args)
