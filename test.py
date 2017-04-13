from NN import *
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

# 创建一个 8x8x1 的神经网络。
network = Network()
network.add_layer(Layer(number_of_neurons=8, input_size=1))
network.add_layer(Layer(number_of_neurons=8))
network.add_layer(Layer(number_of_neurons=1))

# 构造训练数据集：经平移和缩放的正弦曲线。
x = np.arange(0, 2 * np.pi, 0.01)
x = x.reshape((len(x), 1))
y = (np.sin(x) + 1.0) / 2.0

# 绘图。
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)

# 绘制目标曲线。
yt = np.array(y).ravel()
xs = np.array(x).ravel()
ax.plot(xs, yt, label="target", linewidth=2.0)

# 进行 10 个 batch
for i in np.arange(0, 8):
    # 绘制当前网络输出曲线。
    yp = network.predict(x).ravel()
    ax.plot(xs, yp, label="batch {:2d}".format(i))

    print("==================== Batch {:3d} ====================".format(i + 1))

    # 输入 x 为二维数组，形状为 (样本数, 输入向量维度) 。
    # 标准值 y 也是二维数组，形状为 (样本数, 输出向量维度) 。
    network.train(x=x, y=y, eta=0.2, threshold=0.01, max_iters=1000000)

# 训练完成后绘制网络输出曲线。
yp = network.predict(x).ravel()
ax.plot(xs, yp, label="final", linewidth=2.0)


# 目标和输出曲线图。
ax.grid()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim([0, 2 * np.pi])
ax.legend()
plt.savefig("target.png")
plt.clf()
plt.cla()

# MSE 下降图。
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(network.error_history, label="MSE", linewidth=1.5)
ax.legend()
ax.grid()
ax.set_xlabel("epoch")
ax.set_ylabel("MSE")
ax.set_xlim([-5, len(network.error_history) - 1])
plt.savefig("error.png")
plt.clf()
plt.cla()
