import cv2
# TODO 将原视频转换为图片
def VideotoPicture():
    # 视频地址  创建一个VideoCapture对象，指定读取的视频文件
    cap = cv2.VideoCapture(r'D:\DOSC data\VID_20230620_104309.mp4')
    # 通过摄像头的方式
    # cap = cv2.VideoCapture(1)

    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率 每一秒的视频帧数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度

    # 判断视频是否读取成功  成功返回True  失败返回False
    sucess = cap.isOpened()

    frame_count = 0
    # 视频得到的图片名字img_name
    img_name = 0
    while sucess:
        frame_count += 1
        # 读取视频每一帧图像
        sucess, frame = cap.read()
        # TODO 每隔10帧存储一张图片
        if (frame_count % 10 == 0):
            img_name += 1
            # 图片存储地址以及存储格式.jpg  frame就是视频转换得到的图片  视频得到的图片名字img_name从1开始
            # 即最终视频转换得到的图片名字依次为1.jpg，2.jpg，3.jpg.................
            cv2.imwrite(r'D:\DOSC data\picture\%d.jpg' % img_name, frame)

    print("帧率（每秒视频的帧数）:", fps)
    # 释放视频资源
    cap.release


if __name__ == '__main__':
    VideotoPicture()  # 视频转图像
