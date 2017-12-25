# pix2pix-keras-byT 


## 1. prepare the images.

#### move the images to finder like  

images/  
　├  finder1  
　　├  image1.png  
　　├  image2.png  
　　└  image3.png  
　├  finder2  
　　├  image1.png  
　　└ image2.png  
　└ ...  

#### create h5py
if you want to create canny dataset  
`python img2h5.py -i images/ -o datasetimages -t canny`

then, you will have datasetimages.hdf5.


## 2. run.　　

#### run training.  
  
`python pix2pix.py -d datasetimages.hdf5`  

*options

  --datasetpath DATASETPATH, -d DATASETPATH  
  --patch_size PATCH_SIZE, -p PATCH_SIZE  
  --batch_size BATCH_SIZE, -b BATCH_SIZE  
  --epoch EPOCH, -e EPOCH  
  
### generated image.  
'figures' folder will be created in the same directory and generated image would be there.  

generated image sample.  
the upper row is canny image(input image).  
the middle is generated image(output image).  
the lower is original image.  
![generated image sample](http://toxweblog.toxbe.com/wp-content/uploads/2017/12/ss-2017-12-24-12.58.42.png "sample")



