import os

from loguru import logger
import gdown
from huggingface_hub import hf_hub_download
import torch

def dl_adabins(dest=None, is_retry=False):
    logger.debug("Attempting to fetch AdaBins pretrained weights...")
    if not dest:
        dest = os.path.expanduser('~/.cache/adabins/')
    # get your shit together gdown. I should shoot them a PR to fix this silly behavior...
    if not dest.endswith(os.path.sep):
        dest += os.path.sep

    logger.debug(f"using destination path: {dest}")
    #url1 = "https://drive.google.com/uc?id=1lvyZZbC9NLcS8a__YPcUP7rDiIpbRpoF"
    #url2 = "https://drive.google.com/uc?id=1zgGJrkFkJbRouqMaWArXE4WF_rhj-pxW"
    ## if folder does not exist, gdown will create it.
    ## might need to convert folder path to local file system convention. 
    ## gdown checks if path denotes a folder by checking if string terminates with os.path.sep
    ## https://github.com/wkentaro/gdown/blob/main/gdown/download.py#L196-L200
    ## ... yup, this caused a problem. Called it.
    #url = url1 if not is_retry else url2
    #logger.debug(f"downloading from: {url}")
    ## to do: add MD5 hash confirmation
    #response = gdown.download(url, dest)
    #logger.debug(f"gdown response: {response}")
    #return response
    
    response = hf_hub_download(repo_id="deforum/AdaBins", filename="AdaBins_nyu.pt", local_dir=dest)
    logger.debug(f"hf_hub response: {response}")
    return response
    

def save_weights(model, filename, path="./saved_models"):
    if not os.path.isdir(path):
        os.makedirs(path)

    fpath = os.path.join(path, filename)
    torch.save(model.state_dict(), fpath)
    return


def save_checkpoint(model, optimizer, epoch, filename, root="./checkpoints"):
    if not os.path.isdir(root):
        os.makedirs(root)

    fpath = os.path.join(root, filename)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        , fpath)


def load_weights(model, filename, path="./saved_models"):
    fpath = os.path.join(path, filename)
    state_dict = torch.load(fpath)
    model.load_state_dict(state_dict)
    return model


def load_checkpoint(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                           'adaptive_bins_layer.conv3x3.')
            modified[k_] = v
            # del load_dict[k]

        elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    model.load_state_dict(modified)
    return model, optimizer, epoch
