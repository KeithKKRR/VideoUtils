# 个人工具与小技巧
### Video Processing Parallel
代码思路：对于要处理的数据list，使用python脚本，对list等分成N份。每一份分别使用`subprocess.Popen()`开启进程到指定GPU上进行运算。最终等待所有进程结束后移除所有临时文件

特别注意内容：
1) 如有存在视频处理流程内部的临时文件保存，推荐使用`f"file_name_{os.getpid()}.png"`这样的形式来对临时文件进行分开处理。避免进程使用同样的文件名，产生不必要的错误。
2) 进度条设定：暂时没什么好的方案，可以考虑使用多进程中每一个进程都维护自己的进度条，但是使用`subprocess.Popen()`的形式开启进程后，进程并不知道自己应该匹配到哪一个进度条，单开一个参量传递进度条index也没必要。原则上能在终端看到大致进度并且知道什么时候运行结束即可。

参照`my_parallel.py`，使用`--gpu-ids`和`--proc-per-gpu`指定进程数量

### Image Processing Parallel
代码思路：使用torch.data.Dataset处理成batch的形式加速。

特别注意内容：
1) 数据处理的对齐，比如mean、std、size、数据类型等
2) 保留必要信息（如文件名，必要参数等）方便后续保存

参照`my_image_swap.py`

### Face Align, Crop, and Paste-Back
代码思路：
1. DeepfakeBench处理方式：利用五个关键点（左眼、右眼、鼻尖、左嘴角和右嘴角）与预设的五个点的位置计算相似的AffineMatrix，利用这个仿射矩阵对图片进行仿射变换得到对齐裁剪后的图片。对于贴回的话，使用`cv2.invertAffineTransform()`计算相反的仿射矩阵。

特别注意内容： 贴回操作时，由于affine matrix变换的时候会在边缘像素进行插值计算。因此使用像素值计算mask的形式会产生稍微宽一点的mask，导致贴回出现明显黑边。操作就是使用图片的四个顶点计算affine matrix仿射变换回原尺寸的mask，为了避免产生黑边，不能使用最边缘的顶点，需要向内移动2个像素。

2. （尚未解决）其他裁剪模式：第一种方式中当人眼不平行的时候会产生倾斜的矩形，但很多方法针对CelebHQ、FFHQ的数据集进行训练，这类数据集是并不严格与人眼平行的。因此，需要一个横平竖直的裁剪方式。


### JSON Preparation
为了保证操作流程，使用JSON文件对操作流程进行管理，下面是一个sample
```json
[
    {
        "source_path": "video1.mp4",
        "target_path": "video2.mp4",
        "output_path": "video1_video2.mp4",
        "your_param1": 0,
        "your_param2": "something"
    },
    {
        "source_path": "video3.mp4",
        "target_path": "video4.mp4",
        "output_path": "video3_video4.mp4",
        "your_param1": 10,
        "your_param2": "somesomesomething"
    },
    ...
]
```

### Conda Environment
参照`conda pack`的使用流程，对稳定运行的conda环境进行打包备份保存