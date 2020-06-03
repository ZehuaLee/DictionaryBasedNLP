import pickle
import os
from enum import Enum

import argparse
from enum import Enum
import io

from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import matplotlib.pyplot as plt

import difflib
import pandas as pd

import re
import json

import glob

def pickle_dump(obj):
    with open('vision.pickle', 'wb') as f:
        pickle.dump(obj, f)

def pickle_load():
    with open('vision.pickle', 'rb') as f:
        return pickle.load(f)

def pickle_exist():
    os.path.exists('vision.pickle')

class ClassType(Enum):
    NEW = "新設"
    EXISTING = "既設"
    REMOVAL = "撤去"
    FUTURE = "将来"

class SourceType(Enum):
    TOKENIZER = "Tokenizer"
    SIMILAR = "Similar"

class KddiExtraction:
    KEY_WORDS = ["新設", "既設", "撤去", "将来"]
    IGNORE_WORDS = ["EAC殿", "SBM殿", "既設設備(KDDI資産)", "本工事撤去", "他社施工分", "既設流用", "将来用設備"]
    extraction_id = 1
    
    def __init__(self, bound, raw_text):
        self.bound = bound
        self.raw_text = raw_text
        self.is_merged = False
        self.class_type = None
        self.text_color = None
        self.source_type = None
        
        self.model = None
        self.article_name = None
        self.quantity = 0
        
        self.model_desc = None
        self.tokens = []
        
        self.extraction_id = KddiExtraction.extraction_id
        KddiExtraction.extraction_id = KddiExtraction.extraction_id + 1
        
        self.update()
    
    def bound_area(self):
        return "({0},{1}) x ({2},{3})".format(self.bound.vertices[0].x, self.bound.vertices[0].y, self.bound.vertices[3].x, self.bound.vertices[3].y)
    
    def merge(self, a_extraction):
        bound = a_extraction.bound
        raw_text = a_extraction.raw_text
        a_extraction.is_merged = True

        #print("merging", self.raw_text, raw_text)
        self.raw_text = self.raw_text + "\n" + raw_text
        
        if self.bound.vertices[0].x > bound.vertices[0].x: self.bound.vertices[0].x = bound.vertices[0].x
        if self.bound.vertices[0].y > bound.vertices[0].y: self.bound.vertices[0].y = bound.vertices[0].y

        if self.bound.vertices[1].x < bound.vertices[1].x: self.bound.vertices[1].x = bound.vertices[1].x
        if self.bound.vertices[1].y > bound.vertices[1].y: self.bound.vertices[1].y = bound.vertices[1].y

        if self.bound.vertices[2].x < bound.vertices[2].x: self.bound.vertices[2].x = bound.vertices[2].x
        if self.bound.vertices[2].y < bound.vertices[2].y: self.bound.vertices[2].y = bound.vertices[2].y

        if self.bound.vertices[3].x > bound.vertices[3].x: self.bound.vertices[3].x = bound.vertices[3].x
        if self.bound.vertices[3].y < bound.vertices[3].y: self.bound.vertices[3].y = bound.vertices[3].y
        
        self.update()
        
    def has_key(self):
        text = self.raw_text
        if any(elem in text for elem in KddiExtraction.KEY_WORDS):
            return True
        return False

    def has_ignore(self):
        text = self.raw_text
        if any(elem in text for elem in KddiExtraction.IGNORE_WORDS):
            return True
        return False
    
    def update(self):
        if "新設" in self.raw_text:
            self.class_type = ClassType.NEW
        elif "既設" in self.raw_text:
            self.class_type = ClassType.EXISTING
        elif "撤去" in self.raw_text:
            self.class_type = ClassType.REMOVAL
        elif "将来" in self.raw_text:
            self.class_type = ClassType.FUTURE
    
    def is_extracted(self):
        return (self.class_type != None and self.model != None) 
        
    def display(self):
        class_type = self.class_type_str()
        source_type = self.source_type_str()
        return "class: {0}, model: {1}({2}), qty: {3}, color: {4}".format(class_type, self.model, source_type, self.quantity, self.text_color)
    
    def class_type_str(self):
        return 'N/A' if self.class_type == None else self.class_type.value

    def source_type_str(self):
        return 'N/A' if self.source_type == None else self.source_type.value

class KddiRawTextParser:
    KEY_WORDS = ["新設", "既設", "撤去", "将来"]
    
    def __init__(self, debug=True):
        self.init()
        self.kddi_extractions = []
        self.debug = debug
    
    def pre_check(self, bound, text):
        if len(self.buffer_bounds) < 1: 
            return False
        
        orj_up_y = self.buffer_bounds[0].vertices[0].y
        max_up_y = max([elem.vertices[0].y for elem in self.buffer_bounds])
        max_down_y = max([elem.vertices[3].y for elem in self.buffer_bounds])
        target_up_y = bound.vertices[0].y

        last_right_x = self.buffer_bounds[-1].vertices[2].x
        target_left_x = bound.vertices[0].x
        orj_left_x = self.buffer_bounds[0].vertices[0].x
        max_right_x = max([elem.vertices[3].x for elem in self.buffer_bounds])
        
        # near
        #if self.debug: print('orj_y', orj_y, 'max_y', max_y, 'raw_text', self.get_buffer_text(), 'target_y', target_y, 'target_text', text)
        #if abs(target_up_y - max_up_y) > 10 and self.is_split():
        #    return True
        
        # far
        #if abs(target_up_y - max_up_y) > 10 and abs(target_up_y - max_down_y) > 20:
        if target_up_y - max_up_y > 10:
             return True

        if abs(target_left_x - last_right_x) > 20 and not(orj_left_x < target_left_x and target_left_x < max_right_x):
            return True
        
        return False
    
    def parse(self, bound, text):
        # pre check
        if self.pre_check(bound, text):
            self.flush()
            
        # add
        self.buffer_words.append(text)
        self.buffer_bounds.append(bound)
    
    def get_buffer_text(self):
        return "".join(self.buffer_words)
    
    def is_split(self):
        text = self.get_buffer_text()
        if any(elem in text for elem in self.KEY_WORDS):
            return True
        return False
    
    def split(self):
        if self.is_split():
            self.flush()
            
    def flush(self):
        if len(self.buffer_words) == 0: return
        
        raw_text = self.get_buffer_text()
        bound = self.calc_bound()
        
        self.kddi_extractions.append(KddiExtraction(bound, raw_text))
        self.init()
    
    def complete(self):
        self.flush()

        for base_extraction in self.kddi_extractions:
            base_left_x = base_extraction.bound.vertices[0].x
            base_down_y = base_extraction.bound.vertices[3].y
            for target_extraction in self.kddi_extractions:
                if base_extraction.extraction_id == target_extraction.extraction_id: continue
                if target_extraction.is_merged: continue
                
                target_left_x = target_extraction.bound.vertices[0].x
                target_up_y = target_extraction.bound.vertices[0].y
                
                #if base_extraction.raw_text == '・既設GPSアンテナ(残置)': print(target_extraction.raw_text, target_extraction.has_key(), target_up_y, base_down_y, abs(target_up_y - base_down_y), target_left_x, base_left_x, abs(target_left_x - base_left_x))
                #if not target_extraction.has_key() and abs(target_up_y - base_down_y) < 5 and abs(target_left_x - base_left_x) < 8:
                if not target_extraction.has_key() and abs(target_up_y - base_down_y) < 8 and abs(target_left_x - base_left_x) < 15:
                    #if base_extraction.raw_text == '・既設GPSアンテナ(残置)': print("merged") 
                    base_extraction.merge(target_extraction)
                    break
        
        self.kddi_extractions = [e for e in self.kddi_extractions if not e.is_merged]
        #self.kddi_extractions = [e for e in self.kddi_extractions if e.has_key()]
        
        
    def init(self):
        self.buffer_words = []
        self.buffer_bounds = []
        
        #self.buffer_bound = None
    
    def calc_bound(self):
        if len(self.buffer_bounds) == 0: return None

        base_bound = self.buffer_bounds[0]
        for bound in self.buffer_bounds:
            if base_bound.vertices[0].x > bound.vertices[0].x: base_bound.vertices[0].x = bound.vertices[0].x
            if base_bound.vertices[0].y > bound.vertices[0].y: base_bound.vertices[0].y = bound.vertices[0].y

            if base_bound.vertices[1].x < bound.vertices[1].x: base_bound.vertices[1].x = bound.vertices[1].x
            if base_bound.vertices[1].y > bound.vertices[1].y: base_bound.vertices[1].y = bound.vertices[1].y

            if base_bound.vertices[2].x < bound.vertices[2].x: base_bound.vertices[2].x = bound.vertices[2].x
            if base_bound.vertices[2].y < bound.vertices[2].y: base_bound.vertices[2].y = bound.vertices[2].y

            if base_bound.vertices[3].x > bound.vertices[3].x: base_bound.vertices[3].x = bound.vertices[3].x
            if base_bound.vertices[3].y < bound.vertices[3].y: base_bound.vertices[3].y = bound.vertices[3].y
        
        return base_bound


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5
    
def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y], None, color)        
    return image


def draw_raw_text_and_box(image, color, kddi_extractions):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)
    
    #FONTPATH = '/Library/Fonts/Arial Unicode.ttf'
    FONTPATH = 'misc/ArialUnicode.ttf'
    font = ImageFont.truetype(FONTPATH, 10, encoding='utf-8')
    
    for extraction in kddi_extractions:
        #print("detection: ")
        bound = extraction.bound
        draw.polygon([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y], None, color)
        
        #color = 'pink' if extraction.model != None else color
        
        draw.text((bound.vertices[1].x, bound.vertices[0].y), extraction.bound_area(), font=font, fill=color)
        draw.text((bound.vertices[1].x, bound.vertices[0].y+10), extraction.raw_text, font=font, fill=color)
        
        if extraction.is_extracted():
            draw.text((bound.vertices[1].x, bound.vertices[0].y+20), extraction.display(), font=font, fill='pink')
        
        
        print(extraction.bound_area(), extraction.raw_text)
        
    return image

def get_document_bounds(image_file, feature, use_cache_flag=False):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    bounds = []
    #extractions = []
    
    block_bounds = []
    para_bounds = []
    word_bounds = []
    
    #kddi_devices = []
    parser = KddiRawTextParser()
    
    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    
    document = None
    if pickle_exist() and use_cache_flag:
        print("pickle load!!!")
        document = pickle_load()
    else:
        print("accessing Google Vision API!!!")
        response = client.document_text_detection(image=image, image_context={"language_hints": ["ja"]})
        document = response.full_text_annotation
        pickle_dump(document)
    
    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            block_bounds.append(block.bounding_box)
            
            #print(block)
            #print(block.texts)
            parser.flush()
            
            for paragraph in block.paragraphs:
                para_bounds.append(paragraph.bounding_box)
                parser.split()
                
                for word in paragraph.words:
                    word_bounds.append(word.bounding_box)
                    
                    for symbol in word.symbols:
                        parser.parse(symbol.bounding_box, symbol.text)
                        
    parser.complete()
    
    # The list `bounds` contains the coordinates of the bounding boxes.
    return parser.kddi_extractions, block_bounds, para_bounds, word_bounds

 

def analyze_histgram(raw_text, image_crop, debug=False):
    result = None
    
    try:
        img = np.asarray(image_crop.convert("RGB")).reshape(-1,3)
        #plt.hist(img, color=["red", "green", "blue"], histtype="step", bins=128)
        #plt.show()

        if debug:
            print(raw_text)
            plt.imshow(np.asarray(image_crop))
            plt.show()

        b_counter = 0
        r_counter = 0
        for r, g, b in img:
            if r > 240 and g > 240 and b > 240: continue

            if r < 90 and g < 90 and b > 150: 
                if debug: print('blue',r, g, b)
                b_counter += 1
            if r > 150 and g < 90 and b < 90:
                if debug: print('red',r, g, b)
                r_counter += 1

        blue_ratio = b_counter/ len(img)
        red_ratio = r_counter/ len(img)

        
        if blue_ratio > 0.01:
            result = 'blue'
        elif red_ratio > 0.01:
            result = 'red'

        if debug: 
            print('total', len(img), 'blue_ratio', blue_ratio, 'b_counter', b_counter, 'r_counter', r_counter, 'red_ratio', red_ratio)
            print("\n")
    except:
        result = None
        
    return result


def extract_text_data(filein, use_cache=False, debug=False):
    kddi_extractions, block_bounds, para_bounds, word_bounds = get_document_bounds(filein, FeatureType.BLOCK, use_cache)

    image = Image.open(filein)

    #print(extractions)
    #out_image = draw_boxes_with_annotation(image, 'red', extractions)
    
    
    for extraction in kddi_extractions:
        s_x = extraction.bound.vertices[0].x
        s_y = extraction.bound.vertices[0].y
        e_x = extraction.bound.vertices[2].x
        e_y = extraction.bound.vertices[2].y
        image_crop = image.crop((s_x, s_y, e_x, e_y))
        
        result_color = analyze_histgram(extraction.raw_text, image_crop)
        
        extraction.text_color = result_color
        
        if extraction.class_type == None and result_color == 'blue':
            extraction.class_type = ClassType.NEW
        elif extraction.class_type == None and result_color == 'red':
            extraction.class_type = ClassType.REMOVAL
    
    # filter by having key words
    kddi_extractions = [e for e in kddi_extractions if e.has_key()]
    # filter by not having ignore words
    kddi_extractions = [e for e in kddi_extractions if not e.has_ignore()]
    
    
    resolved_counter = 0
    for extraction in kddi_extractions:
        kddi_class, kddi_model, tokens = parse_elements(extraction.raw_text)
        
        if kddi_model != None:
            extraction.model = kddi_model
            extraction.model_desc = 'found by tokenizer'
            extraction.source_type = SourceType.TOKENIZER
            print('found', kddi_model, 'raw_text', extraction.raw_text)
        else:
            if debug: print('not found', 'n/a', 'raw_text', extraction.raw_text)
        
        if extraction.is_extracted():
            resolved_counter += 1
        
        extraction.tokens = tokens
        
    resolved_ratio = 0 if len(kddi_extractions) == 0 else resolved_counter / len(kddi_extractions)
    print('tokenizer', 'resolved_ratio', resolved_ratio, 'total', len(kddi_extractions), 'resolved_counter', resolved_counter)
    
    return kddi_extractions, block_bounds, para_bounds, word_bounds


def render_doc_text(filein, fileout, kddi_extractions, block_bounds, para_bounds, word_bounds, debug=False):
    image = Image.open(filein)
  
    #draw_boxes(image, word_bounds, 'yellow')
    #draw_boxes(image, para_bounds, 'red')
    #draw_boxes(image, block_bounds, 'blue')

    draw_raw_text_and_box(image, 'green', kddi_extractions)

    image.save(fileout)



from janome.tokenizer import Tokenizer
import re

def parse_elements(input_text, debug=False):
    #print("input", input_text)
    input_text = input_text.strip()
    input_text = re.sub('[\n\s]+', ' ', input_text)
    #print("cleansing", input_text)
    
    kddi_div = None
    kddi_model = None
    
    tokenizer = Tokenizer('model_catalog_mod.csv', udic_enc='utf-8')
    words=[]
    for word in tokenizer.tokenize(input_text):
        words.append(word.surface)
        if debug: print(word)
        
        kddi_field=word.part_of_speech.split(",")[2]
        if kddi_field == 'KDDI Div':
            kddi_div = word.surface
        if kddi_field == 'KDDI Model':
            kddi_model = word.surface
        
        #print(kddi_field)
    #print("tokenizer", ", ".join(words))
    #for token in tokenizer.tokenize(input_text):
    #    print(token)
    
    return kddi_div, kddi_model, words



def similar_str(str1):
    columns=["表層形", "左文脈ID", "右文脈ID", "コスト", "品詞", "品詞細分類1", "品詞細分類2", "品詞細分類3", "活用形", "活用型", "原形", "読み", "発音"]
    df = pd.read_csv("model_catalog_mod.csv", names=columns)
    df = df[df['品詞細分類2'] == 'KDDI Model']
    
    score = 0.0
    selected_str = None
    for index, item in df.iterrows():
        str2 = item['表層形']
        s = difflib.SequenceMatcher(None, str1, str2).ratio()

        #print(str1, "<~>", str2)
        #print("match ratio:", s, "\n")
        
        if score < s:
            score = s
            selected_str = str2
    
    return selected_str, score


def find_similar_text(input_text):
    words = re.split('[\s\(\)\[\]\n\r]+', input_text)
    words = [x for x in words if x]
    
    selected_word = None
    high_score = 0.0
    for word in words:
        selected_str, score = similar_str(word)
        #print('input str:', word, 'similar str:', selected_str, 'score:', score)
        if high_score < score:
            high_score = score
            selected_word = selected_str
            
    return selected_word, high_score


def similar_text_data(extractions, threshold_similar_text=0.7, debug=False):
    resolved_counter=0
    total_counter=0
    
    for extraction in extractions:
        if extraction.is_extracted(): continue
        
        total_counter += 1
        #print('tokens', extraction.tokens)
        selected_word, high_score = find_similar_text(extraction.raw_text)
        
        if debug: print('raw_text:', extraction.raw_text, 'selected str:', selected_word, 'score:', high_score)
        if threshold_similar_text < high_score:
            extraction.model = selected_word
            extraction.model_desc = 'found by similar_text'
            extraction.source_type = SourceType.SIMILAR
            resolved_counter += 1
            
            print('similar text', selected_word, 'raw text', extraction.raw_text)
            
    resolved_ratio = 0 if total_counter == 0 else resolved_counter / total_counter
    print('similar text detection', 'resolved_ratio', resolved_ratio, 'total', total_counter, 'resolved_counter', resolved_counter)
    

def extract_regex_quantity(extractions, debug):
    resolved_counter=0
    total_counter=0
    
    for extraction in extractions:
        extraction.quantity = regex_quantity(extraction.raw_text, debug)
    


def regex_quantity(text, debug=False):
    default_quantity_value = 1
    result = re.findall(r'x\s*(\d+)', text)

    if len(result) > 0:
        quantity = int(result[0])
        if debug: print('found by x\s*(\d+)', quantity)
        return quantity
    
    result = re.findall(r'×\s*(\d+)', text)
    if len(result) > 0:
        quantity = int(result[0])
        if debug: print('found by ×\s*(\d+)', quantity)
        return quantity    
    
    return default_quantity_value


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './client_credentials.json'

def runner(image_path, out_path, output_dir, use_cache=False, debug=False, debug_method=True):
    extractions, block_bounds, para_bounds, word_bounds = extract_text_data(image_path, use_cache, debug)

    threshold_similar_text=0.70
    similar_text_data(extractions, threshold_similar_text)
    extract_regex_quantity(extractions, debug)
    
    if debug_method:
        debugging(extractions, image_path, out_path, output_dir)

    render_doc_text(image_path, out_path, extractions, block_bounds, para_bounds, word_bounds, debug)
    
    return extractions, image_path, out_path, output_dir

#import matplotlib.pyplot as plt

def debugging(extractions, image_path, out_path, output_dir):
    result_dict = {
        "input": image_path,
        "output": out_path,
        "crop_list": {}
    }

    data_json_path = output_dir + 'data.json'

    removal_file_list = glob.glob(out_path + "*.png")
    for file in removal_file_list:
        os.remove(file)

    if os.path.exists(data_json_path):
        os.remove(data_json_path)
        
    image = Image.open(image_path)
    
    for index, extraction in enumerate(extractions):
        s_x = extraction.bound.vertices[0].x
        s_y = extraction.bound.vertices[0].y
        e_x = extraction.bound.vertices[2].x
        e_y = extraction.bound.vertices[2].y
        image_crop = image.crop((s_x, s_y, e_x, e_y))
        
        #img = np.asarray(image_crop.convert("RGB")).reshape(-1,3)
        #plt.hist(img, color=["red", "green", "blue"], histtype="step", bins=128)
        #plt.show()

        #plt.imshow(np.asarray(image_crop))
        #plt.show()
        #print(extraction.raw_text)
        #print(extraction.display() + "\n")
        crop_id = str(index+1)
        crop_path = output_dir + crop_id + ".png"
        print(crop_path)
        image_crop.save(crop_path)

        result_dict["crop_list"][crop_id] = {
            "img": crop_path,
            "class": extraction.class_type_str(),
            "model": extraction.model,
            "source": extraction.source_type_str(),
            "quantity": extraction.quantity,
            "raw_text": extraction.raw_text,
            "text_color": extraction.text_color
        }

    f = open(data_json_path, 'w')
    json.dump(result_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))


def execute(input_list, output_base_dir, use_cache=False, debug=False):
    img_list = glob.glob(input_list)

    for img_path in sorted(img_list):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_dir = output_base_dir + basename + '/'
        os.makedirs(output_dir, exist_ok=True)
        #output_inimg_path = output_dir + 'input.png'
        output_outimg_path = output_dir + 'output.png'

        print('img_path', img_path)
        print('output_dir', output_dir)
        print('output_outimg_path', output_outimg_path)
        #print('output_path', output_path)
        runner(img_path, output_outimg_path, output_dir, use_cache, debug)

if __name__ == '__main__':
    use_cache=False
    debug=False

    INPUT_LIST = 'data/input/*.png'
    OUTPUT_DIR = 'data/output/'

    execute(INPUT_LIST, OUTPUT_DIR, use_cache, debug)








