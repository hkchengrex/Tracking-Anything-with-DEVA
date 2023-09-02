import os
from PIL import Image
from multiprocessing import Pool


DIR='imgs'
DIR2='panomasks'

Target_Dir = 'VIPSeg_720P'


def change(DIR,video,image):
    if os.path.isfile(os.path.join(Target_Dir,'images',video,image)) and os.path.isfile(os.path.join(Target_Dir,'panomasks',video,image.split('.')[0]+'.png')):
        return

    img = Image.open(os.path.join(DIR,video,image))
    w,h = img.size
    img = img.resize((int(720*w/h),720),Image.BILINEAR)

    if not os.path.isfile(os.path.join(DIR2,video,image.split('.')[0]+'.png')):
        # print('this is the test set')
        # print(os.path.join(DIR2,video,image.split('.')[0]+'.png'))
        return
    

    mask = Image.open(os.path.join(DIR2,video,image.split('.')[0]+'.png'))
    mask = mask.resize((int(720*w/h),720),Image.NEAREST)

    if not os.path.exists(os.path.join(Target_Dir,'images',video)):
        os.makedirs(os.path.join(Target_Dir,'images',video))
    if not os.path.exists(os.path.join(Target_Dir,'panomasks',video)):
        os.makedirs(os.path.join(Target_Dir,'panomasks',video))

    img.save(os.path.join(Target_Dir,'images',video,image))
    mask.save(os.path.join(Target_Dir,'panomasks',video,image.split('.')[0]+'.png'))
    # print('Processing video {} image {}'.format(video,image))

p = Pool(16)
for video in sorted(os.listdir(DIR)):
    print(video)
    if video[0]=='.':
        continue
    for image in sorted(os.listdir(os.path.join(DIR,video))):
        if image[0]=='.':
            continue
        p.apply_async(change,args=(DIR,video,image))
        #change(DIR,video,image)
p.close()
p.join()
print('finish')

