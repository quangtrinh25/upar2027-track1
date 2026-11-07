"""Portable inference code; dependencies match the official submission runtime."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .model import PARModel
from .calibration import apply_temperature


def preprocess(image,image_size):
    height,width=image_size
    image=image.convert('RGB').resize((width,height),Image.Resampling.BILINEAR)
    pixels=np.asarray(image,dtype=np.float32)/255.
    pixels=(pixels-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(pixels.transpose(2,0,1).copy())


class Predictor:
    def __init__(self,checkpoint,*,device=None,temperature=1.):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.kind=state['kind']
        self.names=state['attribute_names']
        self.image_size=state['image_size']
        self.temperature=float(temperature)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model=PARModel(kind=self.kind,pretrained=False)
        self.model.load_state_dict(state['model'],strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_samples(self,samples,*,batch_size=32):
        if batch_size<1: raise ValueError('batch_size must be positive')
        collected={}
        for start in range(0,len(samples),batch_size):
            batch=samples[start:start+batch_size]
            tensors,orders=[],[]
            for sample in batch:
                names=sample['attribute_names']
                if len(names)!=len(self.names) or set(names)!=set(self.names): raise ValueError('Invalid attribute vocabulary')
                orders.append([self.names.index(name) for name in names])
                with Image.open(sample['image_path']) as image: tensors.append(preprocess(image,self.image_size))
            output=self.model(torch.stack(tensors).to(self.device))
            for key,value in output.items():
                array=value.float().cpu().numpy()
                array=np.take_along_axis(array,np.asarray(orders),axis=1)
                collected.setdefault(key,[]).append(array)
        result={key:np.concatenate(parts) for key,parts in collected.items()}
        if 'probabilities' not in result: raise ValueError('No inference images')
        if self.temperature!=1.: result['probabilities']=apply_temperature(result['probabilities'],self.temperature)
        return result
