import os
import torch
import argparse
import torch

def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r

parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    required=True,
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    required=True,
    help="path to save the converted model",
    type=str,
)

args = parser.parse_args()
#
MODEL_PATH = os.path.expanduser(args.pretrained_path)
print('Model path: {}'.format(MODEL_PATH))

_d = torch.load(MODEL_PATH)
newdict = _d

newdict['model'] = removekey(_d['model'], ['module.roi_heads.box.feature_extractor.fc6.weight',
                                            'module.roi_heads.box.feature_extractor.fc6.bias',
                                            'module.roi_heads.box.feature_extractor.fc7.weight',
                                            'module.roi_heads.box.feature_extractor.fc7.bias',
                                            'module.roi_heads.box.predictor.cls_score.weight',
                                            'module.roi_heads.box.predictor.cls_score.bias',
                                            'module.roi_heads.box.predictor.bbox_pred.weight',
                                            'module.roi_heads.box.predictor.bbox_pred.bias',
                                            'module.roi_heads.mask.feature_extractor.mask_fcn1.weight',
                                            'module.roi_heads.mask.feature_extractor.mask_fcn1.bias',
                                            'module.roi_heads.mask.feature_extractor.mask_fcn2.weight',
                                            'module.roi_heads.mask.feature_extractor.mask_fcn2.bias',
                                            'module.roi_heads.mask.feature_extractor.mask_fcn3.weight',
                                            'module.roi_heads.mask.feature_extractor.mask_fcn3.bias',
                                            'module.roi_heads.mask.feature_extractor.mask_fcn4.weight',
                                            'module.roi_heads.mask.feature_extractor.mask_fcn4.bias',
                                            'module.roi_heads.mask.predictor.conv5_mask.weight',
                                            'module.roi_heads.mask.predictor.conv5_mask.bias',
                                            'module.roi_heads.mask.predictor.mask_fcn_logits.weight',
                                            'module.roi_heads.mask.predictor.mask_fcn_logits.bias'])

newdict = removeKey(_d, ['optimizer', 'scheduler', 'iteration'])

torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))
