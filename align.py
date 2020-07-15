import cv2
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
import os

# 创建mtcnn对象
mtcnn_model = mtcnn()
# 门限函数
threshold = [0.5,0.6,0.6]

def resize_image(image, width, height):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < w:
        dh = w - h
        top = dh // 2
        bottom = dh - top
    else:
        dw = h - w
        left = dw // 2
        right = dw - left
    # else:   #考虑相等的情况（感觉有点多余，其实等于0时不影响结果）
    #     pass
    # RGB颜色
    BLACK = [0, 0, 0]
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回
    constant = cv2.resize(constant, (width, height))
    return constant

def read__image(path_name):
    num = 0
    for dir_image in os.listdir(path_name):  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        full_path = os.path.abspath(os.path.join(path_name, dir_image))

        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read__image(full_path)
        else:  # 如果是文件了
            if dir_image.endswith('.jpg'):
                img = cv2.imread(full_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(full_path)
                # 检测人脸
                rectangles = mtcnn_model.detectFace(img, threshold)

                if rectangles != []:
                    # 转化成正方形
                    rectangles = utils.rect2square(np.array(rectangles))

                    for rectangle in rectangles:
                        if rectangle is not None:
                            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

                            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

                            crop_img = cv2.resize(crop_img,(160,160))
                            # cv2.imshow("before",crop_img)
                            new_img,_ = utils.Alignment_1(crop_img,landmark)
                            # cv2.imshow("two eyes",new_img)
                            cv2.imwrite('C:\\Users\\yu guo long\\PycharmProjects\\face\\mtcnn\\mtcnn-keras-master\\train_data\\2\\' + '%d.jpg' % (
                                    num), new_img)
                            num+=1

            if dir_image.endswith('.jpeg'):
                img = cv2.imread(full_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(full_path)
                rectangles = mtcnn_model.detectFace(img, threshold)

                if rectangles != []:
                    # 转化成正方形
                    rectangles = utils.rect2square(np.array(rectangles))

                    for rectangle in rectangles:
                        if rectangle is not None:
                            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array(
                                [int(rectangle[0]), int(rectangle[1])])) / (rectangle[3] - rectangle[1]) * 160

                            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

                            crop_img = cv2.resize(crop_img, (160, 160))
                            # cv2.imshow("before", crop_img)
                            new_img, _ = utils.Alignment_1(crop_img, landmark)
                            # cv2.imshow("two eyes", new_img)

                            # cv2.waitKey(0)
                            cv2.imwrite('C:\\Users\\yu guo long\\PycharmProjects\\face\\mtcnn\\mtcnn-keras-master\\train_data\\2\\'+'%d.jpg'%(num),new_img)
                            num+=1

if __name__ == '__main__':
    read__image('C:\\Users\\yu guo long\\PycharmProjects\\face\\mtcnn\\mtcnn-keras-master\\train_data\\21')