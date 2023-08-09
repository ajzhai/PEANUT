import copy
import numpy as np

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}
hm3d_names = {0: "chair", 1: "bed", 2: "plant",  3: "toilet",  4: "tv_monitor",  5: "sofa"}
hm3d_to_coco = {0: 0, 
                1: 3, 
                2: 2,
                3: 4,
                4: 5,
                5: 1}
coco_to_hm3d = {v : k for k, v in hm3d_to_coco.items()}
hm3d_to_21 = {
    0: 1,
    1: 7,
    2: 9,
    3: 11,
    4: 14,
    5: 6
}
twentyone_to_hm3d = {v : k for k, v in hm3d_to_21.items()}

hm3d_to_ade = {
    0:19, 
    1:7, 
    2:17, 
    3:65, 
    4:89, 
    5:23
}

scenes = {}
scenes["train"] = [
    'Allensville',
    'Beechwood',
    'Benevolence',
    'Coffeen',
    'Cosmos',
    'Forkland',
    'Hanson',
    'Hiteman',
    'Klickitat',
    'Lakeville',
    'Leonardo',
    'Lindenwood',
    'Marstons',
    'Merom',
    'Mifflinburg',
    'Newfields',
    'Onaga',
    'Pinesdale',
    'Pomaria',
    'Ranchester',
    'Shelbyville',
    'Stockman',
    'Tolstoy',
    'Wainscott',
    'Woodbine',
]

scenes["val"] = [
    'Collierville',
    'Corozal',
    'Darden',
    'Markleeville',
    'Wiconisco',
]


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

habitat_labels_whilelist = [
    {1,2,20,17,12},
    {4,8},
    {5,6},
    {11,15,16},
]

whitelist_map = {
    1:0,
    2:0,
    20:0,
    17:0,
    12:0,
    4:1,
    8:1,
    5:2,
    6:2,
    11:3,
    15:3,
    16:3
}

habitat_labels_r = {
            0:'background',
            1:'chair',
            2:'table',
            3:'picture',
            4:'cabinet',
            5:'cushion',
            6:'sofa',
            7:'bed',
            8:'chest_of_drawers',
            9:'plant',
            10:'sink',
            11:'toilet',
            12:'stool',
            13:'towel',
            14:'tv_monitor',
            15:'shower',
            16:'bathtub',
            17:'counter',
            18:'fireplace',
            19:'gym_equipment',
            20:'seating',
            21:'clothes',
}

hab2coco = {
    1:0,
    2:6,
    6:1,
    7:3,
    9:2,
    11:4,
    14:5,
}
hab2name = {
    1:"chair",
    2:"table",
    6:"sofa",
    7:"bed",
    9:"plant",
    11:"toilet",
    14:"tv",
}

habitat_goal_label_to_similar_coco = {
    1:0, #chair chair
    2:6, #table dining table
    3:12, #picture vase
    4:5, #cabinet tv
    5:1, #cushion couch
    6:1, #sofa couch
    7:3, #bed bed
    8:5, #drawers tv
    9:2, #plant potted plant
    10:8, #sink sink
    11:4, #toilet toilet
    12:1, #stool chair
    13:4, #towel toilet
    14:5, #tv tv
    15:4, #shower toilet
    16:4, #bathtub toilet
    17:6, #counter dining table
    18:5, #fireplace tv
    19:0, #gym chair
    20:0, #seating chair
    21:10, #clothes book
}

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

compatible_dict = {
    2:{8},
    4:{8},
    5:{6,7},
    6:{5,21},
    7:{5},
    8:{2,4},
    10:{11,15,16},
    11:{10,15,16},
    13:{11,15,16},
    15:{11,13},
    16:{11,13},
    17:{8}
}

white_list = {1: {2, 12}, 2: {20}, 3: {13}, 4: {17}, 5: {6}, 20: {2, 18}}
black_list = {2: {1, 4}, 3: {10}, 4: {8}, 8: {4}}

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.9, 0.9, 0.9,  # formerly 0.95
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999,
0.300000000000001, 0.66, 0.8531999999999998,
0.9400000000000001, 0.06, 0.8531999999999998,
0.9400000000000001, 0.66, 0.5531999999999998,
0.3400000000000001, 0.96, 0.2531999999999998,
]

# Pulled from https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv
mpcat40_labels = [
    # '', # -1
    #'void', # 0
    'wall',
    'floor',
    'chair',
    'door',
    'table', # 5
    'picture',
    'cabinet',
    'cushion',
    'window',
    'sofa', # 10
    'bed',
    'curtain',
    'chest_of_drawers',
    'plant',
    'sink',
    'stairs',
    'ceiling',
    'toilet',
    'stool',
    'towel', # 20
    'mirror',
    'tv_monitor',
    'shower',
    'column',
    'bathtub',
    'counter',
    'fireplace',
    'lighting',
    'beam',
    'railing',
    'shelving',
    'blinds',
    'gym_equipment', # 33
    'seating',
    'board_panel',
    'furniture',
    'appliances',
    'clothes',
    'objects',
    'misc',
    'unlabeled' # 41
]

fourty221_ori = {}
twentyone240 = {}
for i in range(len(mpcat40_labels)):
    lb = mpcat40_labels[i]
    if lb in habitat_labels.keys():
        fourty221_ori[i] = habitat_labels[lb]
        twentyone240[habitat_labels[lb]] = i

fourty221 = copy.deepcopy(fourty221_ori)

# catmap = np.loadtxt('Stubborn/category_mapping.tsv', dtype=str,  delimiter='\t')
# raw_name_to_mpcat40 = {}
# for i in range(1, catmap.shape[0]):
#     raw_name_to_mpcat40[catmap[i][1]] = catmap[i][-1]

