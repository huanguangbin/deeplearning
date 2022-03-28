import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def write_csv():
    total_num = 500
    noise_num = int(total_num * 0.01)
    save_path = "./test_waist1.csv"
    add_waistline = True
    featrue_dims = 4 if add_waistline else 3
    columns = ["sex", "height", "weight"]
    man_num = np.random.randint(int(total_num * 0.3), int(total_num * 0.7))
    female_num = total_num - man_num - noise_num
    arrs = np.zeros([featrue_dims, total_num])
    noise_sex_arr = np.random.randint(0, 2, noise_num)
    man_sex_arr = np.ones(man_num)
    female_sex_arr = np.zeros(female_num)
    noise_height_arr = np.random.normal(170, 50, noise_num)
    noise_height_arr[np.where(noise_height_arr < 0)] = 1
    noise_weight_arr = np.random.normal(100, 50, noise_num)
    noise_weight_arr[np.where(noise_weight_arr < 0)] = 1
    man_height_arr = np.random.normal(172, 5, man_num)
    man_weight_arr = np.random.normal(60, 10, man_num)
    female_height_arr = np.random.normal(160, 5, female_num)
    female_weight_arr = np.random.normal(45, 10, female_num)
    sex_arr = np.append(man_sex_arr, female_sex_arr)
    sex_arr = np.append(noise_sex_arr, sex_arr)
    height_arr = np.append(man_height_arr, female_height_arr)
    height_arr = np.append(noise_height_arr, height_arr)
    weight_arr = np.append(man_weight_arr, female_weight_arr)
    weight_arr = np.append(noise_weight_arr, weight_arr)
    arrs[0] = sex_arr
    arrs[1] = height_arr
    arrs[2] = weight_arr
    if featrue_dims == 4:
        columns.append("waistline")
        noise_waistline_arr = np.random.normal(70, 20, noise_num)
        man_waistline_arr = np.random.normal(70, 5, man_num)
        female_waistline_arr = np.random.normal(60, 5, female_num)
        waistline_arr = np.append(man_waistline_arr, female_waistline_arr)
        waistline_arr = np.append(noise_waistline_arr, waistline_arr)
        arrs[3] = waistline_arr
    arrs = arrs.transpose([1, 0])
    frame = pd.DataFrame(arrs, columns=columns)
    frame.to_csv(save_path, index=False)


def read_csv():
    frame = pd.read_csv("test.csv")
    df = pd.DataFrame(frame)
    arr = np.array(df)
    print(arr)


def plot_data():
    delta = 0.02
    frame = pd.read_csv("test.csv")
    df = pd.DataFrame(frame)
    arr = np.array(df)
    # x1_plot = np.linspace(100, 200, 1000)
    # x2_plot = f(x1_plot)
    h_min, h_max = np.min(arr[:, 1]) - 1, np.max(arr[:, 1]) + 1
    w_min, w_max = np.min(arr[:, 2]) - 1, np.max(arr[:, 2]) + 1
    h, w = np.meshgrid(np.arange(h_min, h_max, delta), np.arange(w_min, w_max, delta))
    # plt.contour(h, w, np.full_like(h,180), colors=['blue'])
    plt.contourf(h, w, f(h, w))
    plt.scatter(arr[:, 1], arr[:, 2], c=arr[:, 0])
    # plt.plot(x1_plot,x2_plot)
    pass


def plot_tst():
    delta = 0.5
    frame = pd.read_csv("./test_waist1.csv")
    df = pd.DataFrame(frame)
    arr = np.array(df)
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    h_min, h_max = np.min(arr[:, 1]) - 1, np.max(arr[:, 1]) + 1
    w_min, w_max = np.min(arr[:, 2]) - 1, np.max(arr[:, 2]) + 1
    waist_min, waist_max = np.min(arr[:, 3]) - 1, np.max(arr[:, 3]) + 1
    height, weight = np.meshgrid(np.arange(h_min, h_max, delta), np.arange(w_min, w_max, delta))
    ax1.contourf3D(height,weight,70,aalpha=0.1)
    ax1.scatter3D(arr[:, 1], arr[:, 2], arr[:, 3], c=arr[:, 0])
    plt.show()


def f(x, y):
    return (1 - x / 2 + x ** 3 + y ** 5) * np.exp(-x ** 2 - y ** 2)


if __name__ == "__main__":
    # write_csv()
    # read_csv()
    # plot_data()
    plot_tst()
