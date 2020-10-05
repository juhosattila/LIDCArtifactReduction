import json
import numpy as np


class RadonParams:
    def __init__(self, angles: np.ndarray or int, is_degree=True, projection_width: int = None):
        """
        Args:
            angles: if array, then it specifies exactly the projections angles;
                    if integer, then it specifies the number of projections placed at equal angles
                        from each other in the range 0. to 180. degress.
        """
        self._angles = angles
        if isinstance(self._angles, int):
            self._angles = np.linspace(0.0, 180.0, self._angles)

        self._is_degree = is_degree
        self._projection_width = projection_width

    @property
    def angles(self):
        return self._angles

    @property
    def is_degree(self):
        return self._is_degree

    @property
    def projection_width(self):
        return self._projection_width

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
