class RadonGeometry:
    def __init__(self, volume_img_width, projection_width: int, nr_projections: int):
        """
        Args:
            volume_img_width: the width of volume slice images in number of pixels. Images are squared.
            projection_width: specifies the width of the detector/projection.
            nr_projections: it specifies the number of projections placed at equal angles
                        from each other in the range 0. to 180. degrees.
        """
        self.volume_img_width = volume_img_width
        self.nr_projections = nr_projections
        self.projection_width = projection_width

    def toJson(self):
        return self.__dict__

    @classmethod
    def fromJson(cls, jsons):
        return cls(**jsons)
