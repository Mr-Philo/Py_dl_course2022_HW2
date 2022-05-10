# Py_dl_course2022_HW2


该项目为2022春“Python与深度学习基础”课程大作业项目二
<br>

#### 一、根据Tiny-ImageNet图片大小（3\*64\*64），计算图片经过各层处理后的中间结果的大小
请参考construction.txt文件
<br>

#### 二、在pytorch官方所提供的示例代码main.py的基础上，做必要的改动以使得其可以在Tiny-ImageNet上训练。
请参考change.patch文件
<br>

#### 三、在代码中增加torch.utils.tensorboard的代码以可视化。
#### + 四、运行程序，将resnet18在训练集上的精度（Top5）训练到95%以上。
请参考runlog_20220502文件，其内有在BitaHub上训练时JupyterLab上的屏幕截图和TensorBoard的曲线界面
<br>

#### 五、至少保存2个训练过程中模型的checkpoint，并使用代码中的--evaluate选项，对比两次评估的差异。
请参考evaluate_checkpoints文件夹，其内有评估checkpoints表现时代码运行的屏幕截图，和代码所生成的checkpoints模型对验证集前100张图片的评判结果。
<br>
