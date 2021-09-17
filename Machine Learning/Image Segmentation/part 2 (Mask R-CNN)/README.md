## Mask R-CNN Tutorial for Object Detection and Segmentation

In this Mask R-CNN tutorial, I'm going to provide a tutorial on using [Mask R-CNN](https://github.com/matterport/Mask_RCNN) created by the data scientists and researchers at Facebook AI Research (FAIR).

Since the [Mask R-CNN](https://github.com/matterport/Mask_RCNN) repo has not been updated since 2019, I need to downgrade my system requirements (see the Requirements section below).

## Execution
1. Setting up python 3.6 environment using conda
```
conda create -n python36 python=3.6
```

2. Using python 3.6
```
conda activate python36
```

3. Installing requirements
```
pip install tensorflow-gpu==1.15
pip install keras==2.0.8
pip install Cython
pip install scikit-image
conda install -c conda-forge jupyterlab
```

4. Installing Mask_RCNN
1. `git clone https://github.com/matterport/Mask_RCNN.git`
2. `cd Mask_RCNN`
3. Since we will be using COCO example, we need to copy `coco.py`: `cp samples/coco/coco.py mrcnn/.`
4. `python setup.py install`

## Potential Error

If Keras gives you this error:
```
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-3-138183a2a830> in <module>()
      3 
      4 # Load weights trained on MS-COCO
----> 5 model.load_weights(COCO_MODEL_PATH, by_name=True)

/home/yanxon/anaconda3/envs/python36/lib/python3.6/site-packages/mask_rcnn-2.1-py3.6.egg/mrcnn/model.py in load_weights(self, filepath, by_name, exclude)
   2128 
   2129         if by_name:
-> 2130             saving.load_weights_from_hdf5_group_by_name(f, layers)
   2131         else:
   2132             saving.load_weights_from_hdf5_group(f, layers)

/home/yanxon/anaconda3/envs/python36/lib/python3.6/site-packages/keras/engine/topology.py in load_weights_from_hdf5_group_by_name(f, layers)
   3113     if 'keras_version' in f.attrs:
-> 3114         original_keras_version = f.attrs['keras_version'].decode('utf8')

AttributeError: 'str' object has no attribute 'decode'
```

Try replacing the code with:
```
3113     if 'keras_version' in f.attrs:
3114         try:
3115             original_keras_version = f.attrs['keras_version'].decode('utf8')
3116         except:
3117             original_keras_version = f.attrs['keras_version']
3118     else:
3119         original_keras_version = '1'
3120     if 'backend' in f.attrs:
3121         try:
3122             original_backend = f.attrs['backend'].decode('utf8')
3123         except:
3124             original_backend = f.attrs['backend']
```


### Requirements
- Python 3.6
- TensorFlow 1.15.0
- Keras 2.0.8
- Jupyter Notebook
- Numpy, skimage, scipy, Pillow, cython, h5py
- pycocotools (see [here](https://github.com/matterport/Mask_RCNN/tree/v2.1#installation) for installation)

### Mask_RCNN Repo
```
https://github.com/matterport/Mask_RCNN
```

### Citation
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
