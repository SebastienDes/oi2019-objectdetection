import os
from xml.dom.minidom import Document
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import csv
import cv2
import string

def WriterXMLFiles(filename, img_path, path, box_list, label_list, difficult_list, w, h, d):
    doc = xml.dom.minidom.Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    nodeFilename = doc.createElement('filename')
    nodeFilename.appendChild(doc.createTextNode(img_path))
    root.appendChild(nodeFilename)

    nodesize = doc.createElement('size')
    nodewidth = doc.createElement('width')
    nodewidth.appendChild(doc.createTextNode(str(w)))
    nodesize.appendChild(nodewidth)

    nodeheight = doc.createElement('height')
    nodeheight.appendChild(doc.createTextNode(str(h)))
    nodesize.appendChild(nodeheight)

    nodedepth = doc.createElement('depth')
    nodedepth.appendChild(doc.createTextNode(str(d)))
    nodesize.appendChild(nodedepth)
    root.appendChild(nodesize)

    for (box, label, difficult) in zip(box_list, label_list, difficult_list):
        x_box = [box[0], box[2], box[4], box[6]]
        xmin = min(x_box)
        xmax = max(x_box)
        y_box = [box[1], box[3], box[5], box[7]]
        ymin = min(y_box)
        ymax = max(y_box)
        nodeobject = doc.createElement('object')
        nodename = doc.createElement('name')
        nodename.appendChild(doc.createTextNode(str(label)))
        nodeobject.appendChild(nodename)

        nodedifficult = doc.createElement('difficult')
        nodedifficult.appendChild(doc.createTextNode(difficult))
        nodeobject.appendChild(nodedifficult)

        nodebndbox = doc.createElement('bndbox')
        nodexmin = doc.createElement('xmin')
        nodexmin.appendChild(doc.createTextNode(str(xmin)))
        nodebndbox.appendChild(nodexmin)

        nodeymin = doc.createElement('ymin')
        nodeymin.appendChild(doc.createTextNode(str(ymin)))
        nodebndbox.appendChild(nodeymin)

        nodexmax = doc.createElement('xmax')
        nodexmax.appendChild(doc.createTextNode(str(xmax)))
        nodebndbox.appendChild(nodexmax)

        nodeymax = doc.createElement('ymax')
        nodeymax.appendChild(doc.createTextNode(str(ymax)))
        nodebndbox.appendChild(nodeymax)

        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)
    fp = open(path + filename, 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


def load_annotation(p):
    '''
    Load annotation from the text file
    :param p: text file path
    :return: numpy arrays of polygons, labels, and difficults
    '''
    text_polys = []
    text_tags = []
    text_difficults = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        for line in f.readlines():
            label = 'text'
            x1, y1, x2, y2, x3, y3, x4, y4, label, difficult = line.split(' ')[0:10]
            liste_polys = [x1, y1, x2, y2, x3, y3, x4, y4]
            liste_polys = [int(float(x)) for x in liste_polys]
            text_polys.append(liste_polys)
            text_tags.append(label)
            text_difficults.append(difficult)

        return np.array(text_polys, dtype=np.int32), np.array(text_tags, dtype=np.str), np.array(text_difficults, dtype=np.str)

if __name__ == "__main__":
    txt_path = './labelTxt/'
    xml_path = './labels/'
    img_path = './images/'

    try:
        assert os.path.exists(txt_path)
    except AssertionError:
        print("Erreur : le repertoire des labels texte n'existe pas.")
        exit(1)
    try:
        assert os.path.exists(img_path)
    except AssertionError:
        print("Erreur : le repertoire des images n'existe pas.")
        exit(1)
    
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)

    txts = os.listdir(txt_path)
    non_empty_txts = [txt for txt in txts if os.path.getsize(txt_path+txt) > 0]
    with open("img_set_file.txt", "w") as f:
        for line in non_empty_txts:
            f.write(line.replace('.txt', '') + "\n")
    for count, t in enumerate(non_empty_txts):
        boxes, labels, difficults = load_annotation(os.path.join(txt_path, t))
        xml_name = t.replace('.txt', '.xml')
        img_name = t.replace('.txt', '.png')
        img = cv2.imread(os.path.join(img_path, img_name))
        h, w, d = img.shape

        WriterXMLFiles(xml_name, img_path+img_name, xml_path, boxes, labels, difficults, w, h, d)

        if count % 1000 == 0:
            print(count)
