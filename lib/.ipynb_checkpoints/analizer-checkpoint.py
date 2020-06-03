
import json
import os
import glob

def number_of_inputs(dir='data/input/'):
    return len(glob.glob(dir + "*.png")) + len(glob.glob(dir + "*.jpg")) + len(glob.glob(dir + "*.jpeg"))

def load_output(dir='data/output/0001'):
    json_path = dir + '/' + 'data.json'
    if not os.path.exists(json_path): return None

    json_load = json.load(open(json_path, 'r'))

    return json_load

def load_validation(dir='data/output/0001'):
    json_path = dir + '/' + 'validation.json'
    if not os.path.exists(json_path): return {'must_exist':[], 'should_not_exist':[]}

    json_load = json.load(open(json_path, 'r'))

    return json_load

def save_validation(validation_dict, dir='data/output/0001'):
    json_path = dir + '/' + 'validation.json'

    f = open(json_path, 'w')
    json.dump(validation_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

def add(json_path, class_val, model, quantity):
    if model == None: return
    validation_dict = load_validation(json_path)

    for i in validation_dict['must_exist']:
        if i['class'] == class_val and i['model'] == model and i['quantity'] == quantity: return

    validation_dict['must_exist'].append({
        'class': class_val,
        'model': model,
        'quantity': quantity
    })

    save_validation(validation_dict, json_path)

def remove(json_path, class_val, model, quantity):
    if model == None: return
    validation_dict = load_validation(json_path)

    for i, e in reversed_enumerate(validation_dict['must_exist']):
        if e['class'] == class_val and e['model'] == model and e['quantity'] == quantity: 
            del validation_dict['must_exist'][i]

    validation_dict['must_exist'].append({
        'class': class_val,
        'model': model,
        'quantity': quantity
    })

    save_validation(validation_dict, json_path)

def reversed_enumerate(lst):
    return zip(reversed(range(len(lst))), reversed(lst))


def checker(json_path):
    json = load_output(json_path)
    
    if json == None: return None, None, None, None, None, None, None
    
    input_path = json["input"]
    output_path = json["output"]
    crop_list = json["crop_list"]

    validation_dict = load_validation(json_path)
    must_exist_list = validation_dict['must_exist']
    should_not_exist_list = validation_dict['should_not_exist']

    for v in must_exist_list:
        for crop in crop_list.values():
            if v["model"] == crop["model"] and v["model"] == crop["model"] and v["quantity"] == crop["quantity"]:
                v["validate"] = 'PASS'
                crop["validate"] = 'PASS'

                v["img"] = crop["img"]
                break

    for v in should_not_exist_list:
        for crop in crop_list.values():
            v["validate"] = 'PASS'
            #crop["validate"] = 'PASS'
            if v["model"] == crop["model"] and v["model"] == crop["model"] and v["quantity"] == crop["quantity"]:
                v["validate"] = 'FAIL'
                crop["validate"] = 'FAIL'

                v["img"] = crop["img"]
                break

    validation_pass = len([e for e in must_exist_list if e.get("validate") == 'PASS']) + len([e for e in should_not_exist_list if e.get("validate") == 'PASS'])
    validation_size = len(must_exist_list) + len(should_not_exist_list)
    validation_fail = validation_size - validation_pass
    validation_ratio = round(validation_pass / validation_size * 100) if validation_size > 0 else 0

    crop_pass = len([e for e in list(crop_list.values()) if e.get("validate") == 'PASS'])
    crop_size = len(crop_list)

    crop_extracted_tokenizer = len([e for e in list(crop_list.values()) if e.get("source") == 'Tokenizer'])
    crop_extracted_similar = len([e for e in list(crop_list.values()) if e.get("source") == 'Similar'])
    crop_extracted_size = crop_extracted_tokenizer + crop_extracted_similar
    crop_extracted_ratio = round(crop_extracted_size / crop_size * 100) if crop_size > 0 else 0

    stuts_dict = {
            "validation_pass": validation_pass,
            "validation_fail": validation_fail,
            "validation_size": validation_size,
            "validation_ratio": validation_ratio,
            "crop_pass": crop_pass,
            "crop_size": crop_size,
            "crop_extracted_tokenizer": crop_extracted_tokenizer,
            "crop_extracted_similar": crop_extracted_similar,
            "crop_extracted_size": crop_extracted_size,
            "crop_extracted_ratio": crop_extracted_ratio
        }
    return input_path, output_path, validation_dict, must_exist_list, should_not_exist_list, crop_list, stuts_dict

