import torch
import typing
import os
from torchvision.utils import save_image

class GradientDb:
    class Entry:
        def __init__(self, image_path: str, cam_id: int) -> None:
            self.image_path = image_path
            self.cam_id = cam_id

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        self.recorded_intermediates: typing.List[GradientDb.Entry] = list()
        self.path: str = path
        self.number: int = 0

    def record_intermediate(self, image: torch.Tensor, camera_id: int) -> None:
        path = self.path + f"/out_{self.number:05d}.png"
        save_image(image, path)
        self.recorded_intermediates.append(GradientDb.Entry(path, camera_id))
        self.number = self.number + 1

    def finalize(self):
        dictionary = {
            "type": "gradient_db",
            "version": 1,
            "recorded_intermediates": self.recorded_intermediates,
        }
        torch.save(dictionary, self.path + "/metadata")
        print(f"written gradient database to {self.path}")


def load_gradient_db(path: str) -> GradientDb:
    dictionary = torch.load(path)
    assert dictionary["type"] == "gradient_db"
    assert dictionary["version"] == 1
    db = GradientDb()
    db.recorded_intermediates = dictionary["recorded_intermediates"]
    return db
