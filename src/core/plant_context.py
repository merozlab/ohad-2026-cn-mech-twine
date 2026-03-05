class PlantContext:
    """
    Read-only registry for cross-object resolution.
    Owns no data, just references to Plant objects.
    """

    def __init__(self, plant_registry):
        # plants: dict[exp_num -> Plant]
        self.plant_registry = plant_registry

    def get_plant(self, exp_num):
        return self.plant_registry[exp_num]