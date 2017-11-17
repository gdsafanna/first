## python for_orders.py orders_fields.csv orders.zip results.csv.gz
import sys
import pandas as pd
import gzip
import xml.etree.ElementTree as ET
import lxml.etree as etree
import re
from zipfile import ZipFile
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
parser = etree.XMLParser(encoding="utf-8",recover=True)

attributes={}
columns=[]

def cleaning_ItemShortDesc(string):
    string=re.sub('Network" ','Network ', string)
    string=re.sub('Life" ', 'Life ', string)
    return string

def getting_data(line, mode, header, columns, attributes, result_path):
    try:
        root=ET.fromstring(line, parser=parser)
        order={col:[] for col in columns[4:]}
        for i in root.iter('OrderLine'):
            for key in attributes.keys():
                for item in i.iter(key):
                    for attr in  item.attrib:
                        if attr in columns:
                            if attr not in ['ExtnPromotionId','ExtnPromotionScheme']:
                                if attr in order.keys():
                                    order[attr].append(item.get(attr))
                                else:
                                    order[attr]=[item.get(attr)]
                    if key in ['OrderLine','Item','LinePriceInfo']:
                        for value in attributes[key]:
                            if value not in item.attrib:
                                order[value].append('')
            for i in root.iter('LineCharges'):
                ExtnPromotionId=[]
                ExtnPromotionScheme=[]
                for j in i.iter('Extn'):
                    for attr in j.attrib:
                        if attr in ['ExtnPromotionId', 'ExtnPromotionScheme']:
                            if attr=='ExtnPromotionId':
                                ExtnPromotionId.append(j.get(attr))
                            else:
                                ExtnPromotionScheme.append(j.get(attr))
            order['ExtnPromotionId'].append(ExtnPromotionId)
            order['ExtnPromotionScheme'].append(ExtnPromotionScheme)

        order['OrderDate']=root.get('OrderDate')
        order['OrderNo']=root.get('OrderNo')
        
        #info Promotion
        for item in root.iter('Promotion'):
            for attr in item.attrib:
                if attr in ['PromotionId', 'PromotionType']:
                    order[attr]=item.get(attr)
        pd.DataFrame(order,columns=columns).rename_axis('index_item').to_csv(result_path, mode=mode, header=header, compression='gzip')
    except ValueError as ve:
        print(ve)
        pass

def main():
    f=sys.argv[1]
    z=sys.argv[2]
    d=sys.argv[2][:-4]
    rp=sys.argv[3]
    attributes={}
    columns=[]
    with open(f,'r') as fields:
              for field in fields:
                columns.append(field.strip().split(':')[1])
                if field.strip().split(':')[0] in attributes.keys():
                    attributes[field.strip().split(':')[0]].append(field.strip().split(':')[1])
                else:
                    attributes[field.strip().split(':')[0]]=[field.strip().split(':')[1]]
    with ZipFile(z,'r') as ozip:
        with ozip.open(d) as file:
            mode='w'
            header=True
            order=''
            for line in file:
                line=line.decode('utf-8').strip()
                if line.startswith('<?xml'):
                    order=line
                else:
                    order=order+line
                if order.endswith('</Order>'):
                    getting_data(order,mode, header, columns, attributes, rp)
                    order=''
                    mode='a'
                    header=False
if __name__ == '__main__':
    main()
