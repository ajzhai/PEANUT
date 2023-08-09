import numpy as np

id_color = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
#         [127, 127, 127],
#         [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)

common_cls = ['chair', 'couch', 'potted plant', 'bed', 'toilet', 'tv', 'dining-table', 'oven', 
              'sink', 'refrigerator', 'book', 'clock', 'vase', 'cup', 'bottle']

categories9 = ['chair', 'sofa', 'plant', 'bed', 'toilet', 'tv_monitor',  
               'fireplace', 'bathtub', 'mirror']

habitat_labels = {
            'background': 0,
            'chair': 1, #g
            'table': 2, #g
            'picture':3, #b
            'cabinet':4, # in resnet
            'cushion':5, # in resnet
            'sofa':6, #g
            'bed':7, #g
            'chest_of_drawers':8, #b in resnet
            'plant':9, #g
            'sink':10, #g
            'toilet':11, #g
            'stool':12, #b
            'towel':13, #b in resnet
            'tv_monitor':14, #g
            'shower':15, #b
            'bathtub':16, #b in resnet
            'counter':17, #b isn't this table?
            'fireplace':18,
            'gym_equipment':19,
            'seating':20,
            'clothes':21 # in resnet
}

categories22 = list(habitat_labels.keys())