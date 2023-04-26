import os
import cv2
import numpy as np
import colorsys
import time
import torch
import torch.nn as nn
from PySide2 import QtWidgets
from PySide2.QtCore import QThread, Signal

from PySide2.QtGui import Qt
from PIL import ImageDraw, ImageFont, Image, ImageQt
from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox
from yolo import YOLO, YOLO_ONNX
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog, QTextEdit, QWidget
from PySide2.QtUiTools import QUiLoader
from threading import Thread, Lock
from PySide2.QtGui import QIcon


import warnings

# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 全局变量
global_var = {
    'confidence': 0.5,  # 置信度
    'IoU': 0.3,  # IoU

    'model_path': "model_data/yolov5_s.pth",  # 当前模型类型smlx
    'phi': 's',  # phi smlx
    'yolo': 0,  # 具体的yolo模型对象

    'file_type': 0,  # 待检测文件类型
    'last_file_type': 0,
    'src_img': 0,  # 待检测的原图
    'dst_img': 0,  # 检测完成的图
    'src_frame': 0,  # 待检测的帧
    'dst_frame': 0,  # 检测完成的帧
    'src_frame_cam': 0,
    'dst_frame_cam': 0,
    'start_disp_video': False,
    'stop_disp_video': False,
    'start_disp_cam': False,
    'stop_disp_cam': False,
    'same_type_change_file': False,

    #'last'
    'windows_path': '',  # 检测文件的windows格式路径
    'linux_path': '',  # 检测文件的linux格式路径
    'last_linux_path': ''  # 检测文件的上一个linux格式路径，用于纠正一个bug
}


class MainWindow(QtWidgets.QWidget):

    # init中定义控件对象并connect到对应的slot
    def __init__(self):
        global global_var
        super(MainWindow, self).__init__()
        # self.video_size = QSize(320, 240)
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('./ui/yolov5.ui')

        # 设置图床颜色
        self.ui.src.setStyleSheet('''background: rgba(177, 177, 177, 0.8);
                                           font-family: YouYuan;
                                           font-size: 12pt;
                                           ''')
        self.ui.src.setAlignment(Qt.AlignCenter)  # 设置字体居中现实
        self.ui.dst.setStyleSheet('''background: rgba(177, 177, 177, 0.8);
                                           font-family: YouYuan;
                                           font-size: 12pt;
                                           ''')
        self.ui.dst.setAlignment(Qt.AlignCenter)  # 设置字体居中现实

        self.ui.Confidence_disp.display(0.5)  # 初始化置信度显示值
        self.ui.IoU_disp.display(0.3)  # 初始化IoU显示值
        self.ui.Confidence_thres.actionTriggered.connect(self.confidence_disp)  # 更新置信度显示
        self.ui.IoU_thres.actionTriggered.connect(self.IoU_disp)  # 更新IoU显示

        self.ui.model.currentIndexChanged.connect(self.select_model)  # 更换模型
        self.ui.type.currentIndexChanged.connect(self.select_type)  # 更换检测文件类型

        self.ui.path.clicked.connect(self.change_file)  # 更换文件按钮

        self.ui.run.clicked.connect(self.run)  # 开始检测按钮
        self.ui.stop.clicked.connect(self.stop_detect)  # 停止检测按钮

        self.run_detect_thread = run_detect() # 检测线程类！
        self.run_detect_thread.sig.connect(self.updateLabel)
        global_var['yolo'] = YOLO()
        print('Model loaded!')
        '''
        init_pic = Image.open('imgs/init.jpg')  # 读原图
        init_pic = init_pic.resize((380, 380))  # resize便于显示
        init_pic = ImageQt.toqpixmap(init_pic)  # PIL转QPixMap格式
        self.ui.src.setPixmap(init_pic)  # 显示
        self.ui.dst.setPixmap(init_pic)  # 显示
        self.ui.src.repaint()  # 重新绘制，必须用，否则第二个绘制的图会对前一个有影响
        self.ui.dst.repaint()  # 重新绘制，必须用，否则第二个绘制的图会对前一个有影响
        self.ui.src.show()
        self.ui.dst.show()
        '''
    # 更新置信度和IoU显示
    def confidence_disp(self):
        global global_var
        self.confidence = self.ui.Confidence_thres.value()  # 当前置信度
        global_var['confidence'] = self.confidence * 1.0 / 100
        self.ui.Confidence_disp.display(self.confidence * 1.0 / 100)

    def IoU_disp(self):
        global global_var
        self.IoU = self.ui.IoU_thres.value()  # 当前IoU
        global_var['IoU'] = self.IoU * 1.0 / 100
        self.ui.IoU_disp.display(self.IoU * 1.0 / 100)

    # 更换模型   self.model_path 模型路径
    def select_model(self):
        global global_var
        if self.ui.model.currentIndex() == 0:  # s
            self.model_path = 'model_data/yolov5_s.pth'
            global_var['phi'] = 's'
        elif self.ui.model.currentIndex() == 1:  # m
            self.model_path = 'model_data/yolov5_m.pth'
            global_var['phi'] = 'm'
        elif self.ui.model.currentIndex() == 2:  # l
            self.model_path = 'model_data/yolov5_l.pth'
            global_var['phi'] = 'l'
        elif self.ui.model.currentIndex() == 3:  # x
            self.model_path = 'model_data/yolov5_x.pth'
            global_var['phi'] = 'x'
        global_var['model_path'] = self.model_path
        thread_create_model = Thread(target=self.thread_create_model)
        thread_create_model.start()  # 开始线程
        thread_create_model.join()  # 等待线程结束

    def thread_create_model(self):
        global_var['yolo'] = YOLO()  # 创建模型

    # 更换检测文件类型
    def select_type(self):
        global global_var
        global_var['last_file_type'] = global_var['file_type']
        global_var['file_type'] = self.ui.type.currentIndex()
        if global_var['file_type'] == 2:
            global_var['stop_disp_video'] = True
            QTextEdit.setPlaceholderText(self.ui.path_disp, '')  # 显示路径

    # 更换文件
    def change_file(self):
        global global_var
        global_var['stop_disp_video'] = True
        global_var['stop_disp_cam'] = True
        '''
        linux_path = QFileDialog.getOpenFileName(self,  # 不继承任何父类
                                                    "Open Image",  # 标题
                                                    os.getcwd(),  # 默认路径
                                                    "All Files (*)",  # 默认显示的文件类型
                                                    None,
                                                    QFileDialog.DontUseNativeDialog)
        '''
        linux_path = QFileDialog.getOpenFileName(self,  # 不继承任何父类
                                                    "Open Image",  # 标题
                                                    os.getcwd(),  # 默认路径
                                                    "All Files (*)",  # 默认显示的文件类型
                                                    )

        if linux_path[0] == '':
            pass
        else:
            global_var['linux_path'] = linux_path[0]
            global_var['windows_path'] = linux_path[0].replace('/', '\\')
            #print(global_var['last_file_type'])
            if global_var['linux_path'] != global_var['last_linux_path'] and global_var['file_type'] == 1 and global_var['last_file_type'] == global_var['file_type']:
                global_var['same_type_change_file'] = True
            else:
                global_var['same_type_change_file'] = False
            global_var['last_linux_path'] = global_var['linux_path']
            QTextEdit.setPlaceholderText(self.ui.path_disp, global_var['windows_path'])  # 显示路径

    def run(self):
        if global_var['file_type'] == 0:
            global_var['stop_disp_cam'] = True
            self.run_detect_thread.start()  # 按下之后直接启动线程
        elif global_var['file_type'] == 1:
            if global_var['stop_disp_video']:
                global_var['stop_disp_video'] = False
                global_var['stop_disp_cam'] = True
                self.run_detect_thread.start()  # 按下之后直接启动线程
            else:
                self.run_detect_thread.start() # 按下之后直接启动线程
        elif global_var['file_type'] == 2:
            if global_var['stop_disp_cam']:
                global_var['stop_disp_cam'] = False
                global_var['stop_disp_video'] = True
                self.run_detect_thread.start()  # 按下之后直接启动线程
            else:
                self.run_detect_thread.start()  # 按下之后直接启动线程

    # 停止检测槽函数
    def stop_detect(self):
        if global_var['file_type'] == 1:
            global_var['stop_disp_video'] = True
            global_var['last_file_type'] = global_var['file_type']
        if global_var['file_type'] == 2:
            #global_var['stop_disp_cam'] = True
            pass

    # 画原始图线程
    def draw_src_pic(self):
        src_img = Image.open(global_var['windows_path'])  # 读原图
        src_img = src_img.resize((380, 380))  # resize便于显示
        src_img = ImageQt.toqpixmap(src_img)  # PIL转QPixMap格式
        self.ui.src.setPixmap(src_img)  # 显示
        self.ui.src.repaint()  # 重新绘制，必须用，否则第二个绘制的图会对前一个有影响
        self.ui.src.show()

    # 画检测后的图线程
    def draw_dst_pic(self):
        dst_img = global_var['dst_img'].resize((380, 380))  # resize便于显示
        dst_img = ImageQt.toqpixmap(dst_img)  # PIL转QPixMap格式
        self.ui.dst.setPixmap(dst_img)  # 显示
        self.ui.dst.repaint()
        self.ui.dst.show()

    # 画原视频帧
    def draw_src_vid(self):
        src_frame = global_var['src_frame'].resize((380, 380))  # resize便于显示
        src_frame = ImageQt.toqpixmap(src_frame)  # PIL转QPixMap格式
        self.ui.src.setPixmap(src_frame)  # 显示
        self.ui.src.repaint()
        self.ui.src.show()

    # 画检测后的视频帧
    def draw_dst_vid(self):
        dst_frame = global_var['dst_frame'].resize((380, 380))  # resize便于显示
        dst_frame = ImageQt.toqpixmap(dst_frame)  # PIL转QPixMap格式
        self.ui.dst.setPixmap(dst_frame)  # 显示
        self.ui.dst.repaint()
        self.ui.dst.show()

    # 画原摄像头
    def draw_src_cam(self):
        src_frame_cam = global_var['src_frame_cam'].resize((380, 380))  # resize便于显示
        src_frame_cam = ImageQt.toqpixmap(src_frame_cam)  # PIL转QPixMap格式
        self.ui.src.setPixmap(src_frame_cam)  # 显示
        self.ui.src.repaint()
        self.ui.src.show()

    # 画检测后摄像头
    def draw_dst_cam(self):
        dst_frame_cam = global_var['dst_frame_cam'].resize((380, 380))  # resize便于显示
        dst_frame_cam = ImageQt.toqpixmap(dst_frame_cam)  # PIL转QPixMap格式
        self.ui.dst.setPixmap(dst_frame_cam)  # 显示
        self.ui.dst.repaint()
        self.ui.dst.show()

    # 类更新label
    def updateLabel(self, str_type):
        global global_var
        if str_type == 'pic':
            thread_draw_src_pic = Thread(target=self.draw_src_pic)
            thread_draw_src_pic.start()
            thread_draw_src_pic.join()
            thread_draw_dst_pic = Thread(target=self.draw_dst_pic)
            thread_draw_dst_pic.start()
            thread_draw_dst_pic.join()
        elif str_type == 'video_src' and global_var['start_disp_video']:
            # src
            thread_draw_src_vid = Thread(target=self.draw_src_vid)
            thread_draw_src_vid.start()
            thread_draw_src_vid.join()
        elif str_type == 'video_dst':
            # dst
            thread_draw_dst_vid = Thread(target=self.draw_dst_vid)
            thread_draw_dst_vid.start()
            thread_draw_dst_vid.join()
        elif str_type == 'cam_src' and global_var['start_disp_cam']:
            # src
            thread_draw_src_cam = Thread(target=self.draw_src_cam)
            thread_draw_src_cam.start()
            thread_draw_src_cam.join()
        elif str_type == 'cam_dst':
            # dst
            thread_draw_dst_cam = Thread(target=self.draw_dst_cam)
            thread_draw_dst_cam.start()
            thread_draw_dst_cam.join()


##################################################
'''
# 文件选择错误警告
def no_pic_warning():
    msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'No picture selected!')
    msg_box.exec_()

def no_video_warning():
    msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'No video selected!')
    msg_box.exec_()

def no_capture_warning():
    msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Capture unavailable!')
    msg_box.exec_()
'''


class run_detect(QThread):
    sig = Signal(str)

    def __init__(self):
        super(run_detect, self).__init__()

    def run(self):
        global global_var
        # 图片
        if global_var['file_type'] == 0:
            if global_var['windows_path'] == '':  # 是否有路径
                #no_pic_warning()
                pass
            else:  # 有路径
                file, file_type = os.path.splitext(os.path.basename(global_var['windows_path']))  # 待检测文件类型
                if file_type != '.jpg' and '.png':  # 类型不正确
                    #no_pic_warning()
                    pass
                else:  # 显示
                    global_var['file_type'] = 0
                    global_var['src_img'] = Image.open(global_var['windows_path'])
                    thread_detect_img = Thread(target=self.thread_detect_img)
                    thread_detect_img.start()  # 开始线程
                    thread_detect_img.join()  # 等待线程结束

            # 传参数回去
            self.sig.emit('pic')

        # 视频
        elif global_var['file_type'] == 1:
            if global_var['windows_path'] == '':
                #no_video_warning()
                pass
            else:
                file, file_type = os.path.splitext(os.path.basename(global_var['windows_path']))  # 待检测文件类型
                if file_type != '.mp4':
                    #no_video_warning()
                    pass
                else:
                    ###############################################
                    global_var['file_type'] = 1
                    # 读视频
                    capture = cv2.VideoCapture(global_var['windows_path'])
                    # 循环读取视频/摄像头的每一帧，并检测显示
                    while True:
                        if global_var['stop_disp_video']:
                            if global_var['file_type'] != 1: break  # 断点接着显示
                            if global_var['same_type_change_file']:
                                global_var['same_type_change_file'] = False
                                break
                        else:
                            # 读取某一帧   frame是当前读进来的一帧
                            ref, frame = capture.read()
                            if not ref:
                                break
                            # 格式转变，BGRtoRGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # 转变成Image   PIL格式
                            global_var['src_frame'] = Image.fromarray(np.uint8(frame))
                            self.sig.emit('video_src')
                            thread_detect_video = Thread(target=self.thread_detect_video)
                            thread_detect_video.start()  # 开始线程
                            thread_detect_video.join()  # 等待线程结束
                            global_var['start_disp_video'] = True

                            #time.sleep(0.01)
                            # 传参数回去
                            self.sig.emit('video_dst')
                    global_var['start_disp_video'] = False

        # 摄像头
        elif global_var['file_type'] == 2:
            # 读视频
            capture = cv2.VideoCapture(0)
            # 循环读取视频/摄像头的每一帧，并检测显示
            while True:
                if not global_var['stop_disp_cam']:
                    # 读取某一帧   frame是当前读进来的一帧
                    ref, frame = capture.read()
                    if not ref:
                        break
                    # 格式转变，BGRtoRGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 转变成Image   PIL格式
                    global_var['src_frame_cam'] = Image.fromarray(np.uint8(frame))
                    self.sig.emit('cam_src')
                    thread_detect_cam = Thread(target=self.thread_detect_cam)
                    thread_detect_cam.start()  # 开始线程
                    thread_detect_cam.join()  # 等待线程结束
                    global_var['start_disp_cam'] = True
                    # 传参数回去
                    self.sig.emit('cam_dst')
                else:
                    cv2.destroyAllWindows()
                    break
            global_var['start_disp_cam'] = False

    # 检测图片线程
    def thread_detect_img(self):
        global global_var
        global_var['dst_img'] = global_var['yolo'].detect_image(global_var['src_img'])

    # 检测视频每一帧线程
    def thread_detect_video(self):
        global global_var
        global_var['dst_frame'] = global_var['yolo'].detect_image(global_var['src_frame'])

    # 检测摄像头每一帧线程
    def thread_detect_cam(self):
        global global_var
        global_var['dst_frame_cam'] = global_var['yolo'].detect_image(global_var['src_frame_cam'])


# predict.py
####################################################################
def predict(PIL_img=None):
    global global_var
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在yolo.py_423行左右处的YOLO_ONNX
    # ----------------------------------------------------------------------------------------------------------#
    if global_var['file_type'] == 0:  # 图片
        mode = "predict"
        img_path = global_var['windows_path']
        # print(img_path)
    elif global_var['file_type'] == 1 or 2:  # 视频,摄像头
        mode = "predict"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    count = False
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    if global_var['file_type'] == 1:  # 视频
        video_path = global_var['windows_path']
    elif global_var['file_type'] == 2:  # 摄像头
        video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    # -------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #
    #   heatmap_save_path仅在mode='heatmap'有效
    # -------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    # -------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    # 装载模型
    # if mode != "predict_onnx":
    #    if global_var['if_load_model'] == True: # 如果模型文件变了，则加载新模型
    #        global_var['yolo'] = YOLO()  # 不用ONNX权重，创建模型
    #    else:
    #        pass
    # else:
    #    global_var['yolo'] = YOLO_ONNX()  # 用ONNX权重

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        if global_var['file_type'] == 0:
            image = Image.open(img_path)
        else:
            image = PIL_img
        # 检测图片并显示绘制后结果
        r_image = global_var['yolo'].detect_image(image, crop=crop, count=count)
        global_var['dst_img'] = r_image


    elif mode == "video":
        # 读视频
        capture = cv2.VideoCapture(video_path)
        # 如果需要保存video
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        # 读一帧
        ref, frame = capture.read()
        # 判断是否正确读入
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        # 循环读取视频/摄像头的每一帧，并检测显示
        while (True):
            t1 = time.time()
            # 读取某一帧   frame是当前读进来的一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image   PIL格式
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测  输出为检测结果并绘制好了框
            frame = np.array(global_var['yolo'].detect_image(frame))
            # RGBtoBGR满足opencv显示格式   OPENCV格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)  # 显示fps

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            # 如果需要保存
            if video_save_path != "":
                out.write(frame)
            # 如果按下ESC，退出
            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")


#######################################################yolo.py
class YOLO(object):
    global global_var
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        # "model_path": global_var['model_path'],
        "classes_path": 'model_data/coco_classes.txt',
        # ---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        # ---------------------------------------------------------------------#
        "anchors_path": 'model_data/yolo_anchors.txt',
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [640, 640],
        # ------------------------------------------------------#
        #   backbone        cspdarknet（默认）
        #                   convnext_tiny
        #                   convnext_small
        #                   swin_transfomer_tiny
        # ------------------------------------------------------#
        "backbone": 'cspdarknet',
        # ------------------------------------------------------#
        #   所使用的YoloV5的版本。s、m、l、x
        #   在除cspdarknet的其它主干中仅影响panet的大小
        # ------------------------------------------------------#
        # "phi": global_var['phi'],
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        # "confidence": global_var['confidence'],
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        # "nms_iou": global_var['IoU'],
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        global global_var
        self.__dict__.update(self._defaults)
        # self.confidence = global_var.get_value('confidence')
        # print(self.confidence)
        # self.nms_iou = global_var.get_value('IoU')
        # self.model_path = global_var.get_value('model_path')
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        # 更新显示
        self._defaults["model_path"] = global_var['model_path']
        self._defaults["classes_path"] = 'model_data/coco_classes.txt'
        self._defaults["anchors_path"] = 'model_data/yolo_anchors.txt'
        self._defaults["anchors_mask"] = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self._defaults["input_shape"] = [640, 640]
        self._defaults["backbone"] = 'cspdarknet'
        self._defaults["phi"] = global_var['phi']
        self._defaults["confidence"] = global_var['confidence']
        self._defaults["nms_iou"] = global_var['IoU']
        self._defaults["letterbox_image"] = True
        self._defaults["cuda"] = True
        show_config(**self._defaults)  # 显示配置

    # ---------------------------------------------------#
    #   生成模型  init调用
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        global global_var
        # ---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        # ---------------------------------------------------#
        self.net = YoloBody(self.anchors_mask, self.num_classes, global_var['phi'], backbone=self.backbone,
                            input_shape=self.input_shape)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(global_var['model_path'], map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(global_var['model_path']))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        global global_var
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image,
                                                         conf_thres=global_var['confidence'],
                                                         nms_thres=global_var['IoU'])

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=confidence,
                                                         nms_thres=IoU)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                # ---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                # ---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=confidence, nms_thres=IoU)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)

        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score = np.max(sigmoid(sub_output[..., 4]), -1)
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches=-0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=confidence,
                                                         nms_thres=IoU)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


if __name__ == '__main__':
    app = QApplication([])
    app.setWindowIcon(QIcon('./imgs/logo.png'))
    mainwindow = MainWindow()
    mainwindow.ui.show()
    app.exec_()
