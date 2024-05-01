import cv2

print('done')

import mmdet

import torch
import torch_mlu
from mmcv.ops import sigmoid_focal_loss
x = torch.randn(3, 10).mlu()
x.requires_grad = True
y = torch.tensor([1, 5, 3]).mlu()
w = torch.ones(10).float().mlu()
output = sigmoid_focal_loss(x, y, 2.0, 0.25, w, 'none')
print(output)


category_info = self.imshow_det_bboxes(img, bboxes, labels, segms, attrs, class_names=self.CLASSES, 
                            score_thr=score_thr, bbox_color=bbox_color, text_color=text_color,
                            mask_color=mask_color, thickness=thickness, font_size=font_size, 
                            win_name=win_name, show=show, wait_time=wait_time,out_file=out_file)

        if not (show or out_file):
            return img, category_info
        else:
            return category_info

            

category_info = []
label_text = f'{class_names[label]} {bbox[-1]*100:.0f}%'
            attr_text = ''
            for idx in attrs[i].nonzero()[0]:
                attr_text = attr_text +  self.attr_classes[idx] + '\n'
            category_info.append([label_text, attr_text])  # Add label and empty attribute for now
return category_info