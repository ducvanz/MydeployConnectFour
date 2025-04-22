import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Input layer
input_layer = layers.Input(shape=(6, 7, 1))
x = layers.Conv2D(128, 3, padding='same')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# Các khối residual
for _ in range(10):
    shortcut = x
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

# Phần đầu chia sẻ
head = layers.Conv2D(32, 1)(x)
head = layers.BatchNormalization()(head)
head = layers.ReLU()(head)
head_flat = layers.Flatten()(head)

# Đầu policy
policy = layers.Dense(7, activation='softmax', name='policy')(head_flat)

# Tạo mô hình với chỉ đầu ra policy
model = models.Model(inputs=input_layer, outputs=policy)

# Biên dịch mô hình
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Hàm tăng cường dữ liệu (flip ngang)
def ss_data(X, y):
    X_flipped = np.flip(X, axis=2)
    y_flipped = np.flip(y, axis=1)
    X_augmented = np.concatenate((X, X_flipped), axis=0)
    y_augmented = np.concatenate((y, y_flipped), axis=0)
    return X_augmented, y_augmented

# Hàm lọc dữ liệu trùng
def filter_data(train_data, labels):
    filtered_train = []
    filtered_labels = []
    seen = set()
    for i in range(len(train_data)):
        hashable = tuple(train_data[i].flatten())
        if np.sum(labels[i] == 1) > 1:
            continue
        if hashable not in seen:
            seen.add(hashable)
            filtered_train.append(train_data[i])
            filtered_labels.append(labels[i])
    return np.array(filtered_train), np.array(filtered_labels)

# Tải dữ liệu
data = np.load("C:/Users/Admin/IdeaProjects/git-connect4/ConnectFour/DL/data/data_MO_MO.npz")
train1 = data["X"]
label1 = data["y"]
data = np.load("C:/Users/Admin/IdeaProjects/git-connect4/ConnectFour/DL/data/data_MO_MO_2.npz")
train2 = data["X"]
label2 = data["y"]
data = np.load("C:/Users/Admin/IdeaProjects/git-connect4/ConnectFour/DL/data/data_MO_MO_3.npz")
train3 = data["X"]
label3 = data["y"]
data = np.load("C:/Users/Admin/IdeaProjects/git-connect4/ConnectFour/DL/data/data_MO_MO_4.npz")
train4 = data["X"]
label4 = data["y"]
data = np.load("C:/Users/Admin/IdeaProjects/git-connect4/ConnectFour/DL/data/data_MO_MO_6.npz")
train5 = data["X"]
label5 = data["y"]
data = np.load("C:/Users/Admin/IdeaProjects/git-connect4/ConnectFour/DL/data/data_MO_MO_7.npz")
train6 = data["X"]
label6 = data["y"]

train = np.concatenate((train1, train2), axis=0)
label = np.concatenate((label1, label2), axis=0)
train = np.concatenate((train, train3), axis=0)
label = np.concatenate((label, label3), axis=0)
train = np.concatenate((train, train4), axis=0)
label = np.concatenate((label, label4), axis=0)
train = np.concatenate((train, train5), axis=0)
label = np.concatenate((label, label5), axis=0)
train = np.concatenate((train, train6), axis=0)
label = np.concatenate((label, label6), axis=0)


train, label = ss_data(train, label)
train, label = filter_data(train, label)

indices = np.random.permutation(train.shape[0])

train = train[indices]
label = label[indices]

train = train.reshape((-1, 6, 7, 1))
print(train.shape)

# Huấn luyện
history = model.fit(
    x=train,
    y=label,
    batch_size=128,
    epochs=20,
    verbose=1
)

# Lưu mô hình
model.save("DL/Files/mymodel22.keras")
