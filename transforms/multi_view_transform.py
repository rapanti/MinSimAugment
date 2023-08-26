from typing import Any, List, Tuple, Union

from PIL.Image import Image
from torch import Tensor


class MultiViewTransform:

    def __init__(self, transforms: List, return_params: bool = False):
        self.transforms = transforms
        self.return_params = return_params

    def __call__(self, image: Union[Tensor, Image]) -> Tuple[Any, Any] | List[Any]:
        inter = [transform(image, return_params=self.return_params) for transform in self.transforms]
        if self.return_params:
            return *[i[0] for i in inter], *[i[1] for i in inter]
        return inter
