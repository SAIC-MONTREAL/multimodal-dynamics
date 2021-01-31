"""
General configs.
"""

# Simulator parameters
TIME_STEP = 1. / 240.
RENDERS = True

# custom datasets in `graphics/`
SHAPENET_DATASETS = ['core', 'sem']

# ShapeNetCore categories with small sized objects suitable for tabletop scenario
SHAPENET_CORE = {
        'bottle': '02876657',
        'bowl': '02880940',
        'camera': '02942699',
        'can': '02946921',
        'cap': '02954340',
        'clock': '03046257',
        'earphone': '03261776',
        'jar': '03593526',
        'knife': '03624134',
        'mug': '03797390',
        'remote': '04074963',
        'telephone': '04401088',
    }

# custom objects in `graphics/objects`
OBJECTS = ['winebottle', 'bowl']

# ShapeNetSem categories with small sized objects suitable for tabletop scenario,
# these are handpicked from the taxanomy.txt accompanying the dataset
DEFAULT_WEIGHT = 1.0
DEFAULT_UNIT = 0.02
DEFAULT_UP = '0\,0\,1'
DEFAULT_FRONT = '1\,0\,0'
COM_THRESHOLD = 2
SHAPENET_SEM = {
    'FoodItem': ['FruitBowl', 'CerealBox', 'Chocolate', 'Cookie', 'MilkCarton', 'Pizza', 'Donut', 'Fruit', 'Sandwich',
                 'Apple', 'Orange', 'Carrot'],

    'Battery': ['AAABattery', 'AABattery'],
    'Bowl': [],
    'Calculator': [],
    'Camera': ['DSLRCamera', 'WebCam'],
    'CanOpener': [],
    'Candle': [],
    'Cap': [],
    'Cassette': [],
    'Coin': [],
    'ComputerMouse': [],
    'Controller': [],
    'DrinkingUtensil': ['Teacup', 'WineGlass', 'Cup', 'Mug'],
    'Eraser': [],
    'Fork': [],
    'Glasses': [],
    'Hammer': [],
    'Hat': [],
    'Headphones': [],
    'Kettle': [],
    'Knife': [],
    'Magnet': [],
    'MediaPlayer': [],
    'Book': [],
    'Books': [],
    'Notepad': [],
    'Pan': [],
    'PaperClip': [],
    'Phone': ['Telephone', 'CellPhone'],
    'PillBottle': [],
    'Ring': [],
    'ScrewDriver': [],
    'Scissors': [],
    'Shampoo': [],
    'SoapBar': [],
    'Spoon': [],
    'SodaCan': [],
    'Stapler': [],
    'Teapot': [],
    'TissueBox': [],
    'ToiletPaper': [],
    'USBStick': [],
    'Vase': [],
    'Watch': [],
    # 'Bottle': ['WineBottle', 'DrinkBottle', 'BeerBottle'],
    'WineBottle': [],
}
