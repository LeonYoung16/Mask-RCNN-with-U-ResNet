import sys
sys.path.append("/home/leon/Desktop/mask/cocoapi/PythonAPI")
from pycocotools.coco import COCO
import numpy as np
#import skimage.io as io
import cv2
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/home/leon/Desktop/mask/maskrcnn-benchmark/datasets/coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'
]);
print(catIds)
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [76846])

img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

print(img)
I = cv2.imread('/home/leon/Desktop/mask/maskrcnn-benchmark/datasets/coco/train2017/'+str(img['file_name']))     
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(I)
plt.show()
print(img['id'])
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
print(annIds)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()

