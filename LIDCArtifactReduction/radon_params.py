import json
import numpy as np


class RadonParams:
    def __init__(self, angles: np.ndarray, is_degree=True, projection_width: int = None):
        self.angles = angles
        self.is_degree = is_degree
        self.projection_width = projection_width

    def toJson(self):
        return {
            'angles': json.dumps(self.angles.tolist()),
            'is_degree': self.is_degree,
            'projection_width': self.projection_width
        }

    @classmethod
    def fromJson(cls, jsons):
        return cls(angles=np.array((jsons['angles'])),
                   is_degree=jsons['is_degree'],
                   projection_width=jsons['projection_width'])
