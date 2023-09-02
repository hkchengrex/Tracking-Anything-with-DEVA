import numpy as np
from deva.utils.vipseg_categories import VIPSEG_CATEGORIES

vipseg_cat_to_isthing = {d['id']: d['isthing'] == 1 for d in VIPSEG_CATEGORIES}


def id_to_rgb(id: np.ndarray) -> np.ndarray:
    h, w = id.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(3):
        rgb[:, :, i] = id % 256
        id = id // 256

    return rgb


class ID2RGBConverter:
    def __init__(self):
        self.all_id = []
        self.obj_to_id = {}

    def _id_to_rgb(self, id: int):
        rgb = np.zeros((3, ), dtype=np.uint8)
        for i in range(3):
            rgb[i] = id % 256
            id = id // 256
        return rgb

    def convert(self, obj: int):
        if obj in self.obj_to_id:
            id = self.obj_to_id[obj]
        else:
            while True:
                id = np.random.randint(255, 256**3)
                if id not in self.all_id:
                    break
            self.obj_to_id[obj] = id
            self.all_id.append(id)

        return id, self._id_to_rgb(id)


class IDPostprocessor:
    def __init__(self):
        self.all_id = []
        self.thing_obj_to_id = {}
        self.stuff_to_id = {}

    def id_to_rgb(self, id):
        rgb = np.zeros((3, ), dtype=np.uint8)
        for i in range(3):
            rgb[i] = id % 256
            id = id // 256
        return rgb

    def _find_new_id(self, default_id):
        id = default_id
        while True:
            if id not in self.all_id:
                return id
            id = np.random.randint(256, 256**3)

    def convert(self, obj, category_id, isthing):
        if isthing:
            # is thing
            if (obj, category_id) in self.thing_obj_to_id:
                id = self.thing_obj_to_id[(obj, category_id)]
            else:
                id = self._find_new_id(obj)
                self.thing_obj_to_id[(obj, category_id)] = id
                self.all_id.append(id)
        else:
            # is stuff
            if category_id in self.stuff_to_id:
                id = self.stuff_to_id[category_id]
            else:
                id = self._find_new_id(obj)
                self.stuff_to_id[category_id] = id
                self.all_id.append(id)

        return id
