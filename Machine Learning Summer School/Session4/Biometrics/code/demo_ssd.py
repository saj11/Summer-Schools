# Just a basic toy example from: https://gluon-cv.mxnet.io
from gluoncv import model_zoo, data, utils
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Load a pretrained model
net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

# Load images
im_fname = 'street_small.jpg'
im_fname = 'ticos.jpg'

x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

# Predict
class_IDs, scores, bounding_boxs = net(x)

# Plot detections & save to a file
ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],	
                         class_IDs[0], class_names=net.classes)
plt.show()
plt.savefig('result.png')