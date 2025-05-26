import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_curve(data, threshold_list):
    x = []
    y = []
    sorted_data = np.sort(data)
    for t in threshold_list:
        y_index = np.where(sorted_data > t)[0][0]
        x.append(sorted_data[y_index])
        y.append(y_index / len(data))
    plt.plot(x, y)


def plot_pixel_mse_curve(data, threshold_list, show_text=False, color="k", legend_name="mse"):
    x = []
    y = []
    scatter_x = []
    scatter_y = []
    sorted_data = np.sort(data)
    scatter_threshold = [0.5, 0.75, 0.8, 0.9, 0.95]
    for t in threshold_list:
        tmp = np.where(sorted_data > t)
        if len(tmp[0]) == 0:
            break
        y_index = tmp[0][0]
        x.append(sorted_data[y_index])
        percentage = y_index / len(data)
        y.append(percentage)
        while len(scatter_threshold) > 0 and percentage > scatter_threshold[0]:
            scatter_index = int(len(data)*scatter_threshold[0])
            x.insert(-1, sorted_data[scatter_index])
            y.insert(-1, scatter_threshold[0])
            scatter_x.append(sorted_data[scatter_index])
            scatter_y.append(scatter_threshold[0])
            scatter_threshold.pop(0)
    plt.plot(x, y, color=color, label=legend_name)
    plt.scatter(scatter_x, scatter_y, color=color, s=10, alpha=0.5)
    # show scatter points position
    if show_text:
        for i in range(len(scatter_x)):
            plt.text(scatter_x[i], scatter_y[i], f"({scatter_x[i]:.2f}, {scatter_y[i]:.2f})", fontsize=10, color=color,
                     alpha=0.5)
    # plot straight line for scatter_threshold in --, in the bottom of the picture
    for i in range(len(scatter_y)):
        plt.plot([0, scatter_x[i]], [scatter_y[i], scatter_y[i]], color='whitesmoke', linestyle='--', zorder=0, alpha=0.5)


    # set x-lim start from 0
    plt.xlim(min(x), max(x))

    # set x label mse in pixel
    # set y label percentage
    plt.xlabel("MSE in pixel (512 x 512)")
    plt.ylabel("Percentage")
    plt.legend()


if __name__ == "__main__":
    # 用法示例
    data_path = "/mnt/maui/CTA_Coronary/project/xiongxs/CTA_DSA/new_pipeline_for_simulated_evaluation/online_cut_sample_single0_lax1_rigid_41_continue_2/t_0.1w_-1d_-1/results.xlsx"

    data = pd.read_excel(data_path)
    threshold_list = np.arange(0, 10, 0.5)
    plot_pixel_mse_curve(np.array(data["mse_opt"]), threshold_list, show_text=True, color="k", legend_name="optimized mse")
    plot_pixel_mse_curve(np.array(data["mse_cut_opt"]), threshold_list, color="skyblue", legend_name="optimized mse_cut")
    plt.xlabel("MSE in pixel (512 x 512)")
    plt.ylabel("Percentage")
    plt.title("Optimized MSE and MSE in cut branch")
    plt.show()

    threshold_list = np.arange(0, 5, 0.1)
    plot_pixel_mse_curve(np.array(data["angle_dis_opt"]), threshold_list, show_text=True, color="k", legend_name="optimized angle")
    plt.xlabel("Rotation in degree")
    plt.ylabel("Percentage")
    plt.title("Optimized Angle Distance")
    plt.savefig("angle_distance.png", pad_inches=0)
    plt.show()

    pass