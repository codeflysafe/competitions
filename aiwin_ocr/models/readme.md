# Model

此文件夹写一些深度学习的模型代码

### CTC-loss

3.2 blank空白标签一定要依据空白符在预测总字符集中的位置来设定，否则就会出错；

3.3 targets建议将其shape设为(sum(target_lengths))，然后再由target_lengths进行输入序列长度指定就好了，这是因为如果设定为(N, S)，则因为S的标签长度如果是可变的，那么我们组装出来的二维张量的第一维度的长度仅为min(S)将损失一部分标签值（多维数组每行的长度必须一致），这就导致模型无法预测较长长度的标签；

3.4 输出序列长度T尽量在模型设计时就要考虑到模型需要预测的最长序列，如需要预测的最长序列其长度为I，则理论上T应大于等于2I+1，这是因为CTCLoss假设在最坏情况下每个真实标签前后都至少有一个空白标签进行隔开以区分重复项；

3.5 输出的log_probs除了进行log_softmax()处理再送入CTCLoss外，还必须要调整其维度顺序，确保其shape为(T, N, C)！