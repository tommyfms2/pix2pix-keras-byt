
import numpy as np
import glob
import argparse
import h5py

from keras.preprocessing.image import load_img, img_to_array
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', '-i', required=True)
    parser.add_argument('--outpath', '-o', required=True)
    parser.add_argument('--trans', '-t', default='gray')
    args = parser.parse_args()

    finders = glob.glob(args.inpath+'/*')
    print(finders)
    imgs = []
    gimgs = []
    for finder in finders:
        files = glob.glob(finder+'/*')
        for imgfile in files:
            img = load_img(imgfile)
            imgarray = img_to_array(img)
            imgs.append(imgarray)
            if args.trans=='gray':
                grayimg = load_img(imgfile, grayscale=True)
                grayimgarray = img_to_array(grayimg)
                gimgs.append(grayimgarray)
            elif args.trans=='canny':
                grayimg = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2GRAY)
                gray_canny_xy = cv2.Canny(grayimg, 128,128 )
                gray_canny_xy = cv2.bitwise_not(gray_canny_xy)
                gimgs.append(gray_canny_xy.reshape(128,128,1))                
                    

    perm = np.random.permutation(len(imgs))
    imgs = np.array(imgs)[perm]
    gimgs = np.array(gimgs)[perm]
    threshold = len(imgs)//10*9
    vimgs = imgs[threshold:]
    vgimgs = gimgs[threshold:]
    imgs = imgs[:threshold]
    gimgs = gimgs[:threshold]
    print('shapes')
    print('gen imgs : ', imgs.shape)
    print('raw imgs : ', gimgs.shape)
    print('val gen  : ', vimgs.shape)
    print('val raw  : ', vgimgs.shape)

    outh5 = h5py.File(args.outpath+'.hdf5', 'w')
    outh5.create_dataset('train_data_gen', data=imgs)
    outh5.create_dataset('train_data_raw', data=gimgs)
    outh5.create_dataset('val_data_gen', data=vimgs)
    outh5.create_dataset('val_data_raw', data=vgimgs)
    outh5.flush()
    outh5.close()


if __name__=='__main__':
    main()
