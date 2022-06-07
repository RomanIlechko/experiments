import xmltodict
import os
from glob import glob
from pathlib import Path
from dict2xml import dict2xml


def xml_(dct):
    return dict2xml(dct, indent="   ")


def read(path):
    with open(path, 'r') as f:
        return f.read()

# s
def write(path, data):
    with open(path, 'w') as f:
        f.write(data)


def all2bdd(dct):
    copy = dct.copy()
    match = {'vehicle': 'car',
             'traffic_sign': 'traffic sign',
             'traffic_light': 'traffic light',
             'motobike': 'moto',
             'bike': 'bike'}
    copy['name'] = match[dct['name']]
    return copy


def arch2bdd(dct):
    copy = dct.copy()
    match = {'passenger_car': 'car',
             'walker': 'person',
             'motorcycle': 'motor',
             'motobike': 'moto',
             'bicycle': 'bike',
             'truck': 'truck'}
    copy['name'] = match[dct['name']]
    return copy


def fix(dct):
    copy = dct.copy()
    if dct['name'] == 'moto':
        copy['name'] = 'motor'
    return copy


if __name__ == "__main__":
    root_path = Path("/Users/romanilechko/MS_UCU/Third_term/diploma/experiment/Carla-Object-Detection-Dataset/full_aug/")
    new_root_path = Path('/Users/romanilechko/MS_UCU/Third_term/diploma/experiment/Carla-Object-Detection-Dataset/xml')
    df = 'all' # 'all' 'arch'
    cnv = fix #{'all': all2bdd, 'arch': arch2bdd}[df]
    xml_paths = glob(os.path.join(root_path, "*.xml"))

    all_cls = dict()
    dl = []
    #write(f'b.txt', '\n'.join(xml_paths))

    for xml_path in xml_paths:
        xml_data = read(xml_path)
        dct = xmltodict.parse(xml_data)
        try:
            objects = dct['annotation']['object']
        except:
            print(xml_path)
            dl.append(f'rm {xml_path}')
            print(f'{Path(xml_path).parent}/{Path(xml_path).stem}.jpg')
            dl.append(f'rm {Path(xml_path).parent}/{Path(xml_path).stem}.jpg')
            continue
        '''
        if isinstance(objects, dict):
            dct['annotation']['object'] = cnv(dct['annotation']['object'])
        else:
            dct['annotation']['object'] = [cnv(obj) for obj in objects]

        res = xml_(dct)
        write(f'{new_root_path}/{Path(xml_path).stem}.xml', res)'''
    write('rm.sh', '\n'.join(dl))

"""
/Users/romanilechko/Downloads/archive/test:                                                  ['passenger_car', 'walker', 'motorcycle', 'bicycle', 'truck']
/Users/romanilechko/Downloads/archive/train:                                                 ['bicycle', 'passenger_car', 'walker', 'motorcycle', 'truck']
/Users/romanilechko/MS_UCU/Third_term/diploma/experiment/Carla-Object-Detection-Dataset/all: ['vehicle', 'traffic_sign', 'bike', 'traffic_light', 'motobike'] # have empty

nice -> /Users/romanilechko/MS_UCU/Third_term/diploma/experiment/data/bdd100k/xml/train:     ['car', 'traffic sign', 'traffic light', 'truck', 'person', 'bus', 'bike', 'rider', 'motor', 'train']

"""
