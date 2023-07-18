import torch
import typing
import os
from torchvision.utils import save_image

class GradientDb:
    class Entry:
        def __init__(self, image_path: str, cam_id: int) -> None:
            self.image_path = image_path
            self.cam_id = cam_id

    def __init__(self, path: str = None) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        self.entries: typing.List[GradientDb.Entry] = list()
        self.path: str = path
        self.number: int = 0

    def load(self):
        dictionary = torch.load(self.path + "/metadata")
        assert dictionary["type"] == "gradient_db"
        assert dictionary["version"] == 1
        self.entries = dictionary["recorded_intermediates"]

    def record_intermediate(self, image: torch.Tensor, camera_id: int) -> None:
        path = self.path + f"/out_{self.number:05d}.png"
        save_image(image, path)
        self.entries.append(GradientDb.Entry(path, camera_id))
        self.number = self.number + 1

    def finalize(self):
        dictionary = {
            "type": "gradient_db",
            "version": 1,
            "recorded_intermediates": self.entries,
        }
        torch.save(dictionary, self.path + "/metadata")
