#from infer import InferenceHelper
from adabins.infer import InferenceHelper


from PIL import Image
from loguru import logger
import matplotlib.pyplot as plt
from einops import rearrange
import torch

infer_helper = InferenceHelper(dataset='nyu')

img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)

im = torch.tensor(predicted_depth)
im = im.squeeze(0)
im2 = rearrange(im, "c h w -> h w c")

logger.debug(f"{predicted_depth.shape} | {im.shape} | {im2.shape}")
plt.imshow(im2)
plt.show()