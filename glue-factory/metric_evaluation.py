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

def plot_cumulated_curve(data, show_text=False, color="k", legend_name="mse"):
    scatter_x = []
    scatter_y = []
    sorted_data = np.sort(data)
    x = sorted_data
    y = (np.arange(len(sorted_data)) + 1) / len(sorted_data)

    scatter_threshold = [0.5, 0.75, 0.8, 0.9, 0.95]
    for t in scatter_threshold:
        scatter_index = int(len(sorted_data) * t)
        scatter_x.append(sorted_data[scatter_index])
        scatter_y.append(y[scatter_index])

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
    extend = 0.1 * (max(x) - min(x))
    plt.xlim(min(x), max(x) + extend)
    plt.legend()



if __name__ == "__main__":
    # 用法示例
    from pathlib import Path
    save_path = Path("path/to/save/plots")
    data_path = Path("path/to/your/infer_results.xlsx")


    data = pd.read_excel(data_path)

    difficult_index = [i for i in range(data['name'][:-1].shape[0]) if 'AI98' in data['name'][i]]
    easy_index = [i for i in range(data['name'][:-1].shape[0]) if 'AI98' not in data['name'][i]]

    plot_cumulated_curve(np.array(data["mse_grid"])[:-1], color="skyblue", legend_name="grid mse")
    plot_cumulated_curve(np.array(data["mse_opt"])[:-1], show_text=True, color="k", legend_name="optimized mse")
    plt.xlabel("MSE in pixel (512 x 512)")
    plt.ylabel("Percentage")
    plt.title("MSE")
    plt.savefig(save_path / "mse.png", pad_inches=0)
    plt.show()

    plot_cumulated_curve(np.array(data["angle_dis_grid"])[:-1], show_text=True, color="skyblue",
                         legend_name="grid angle")
    plot_cumulated_curve(np.array(data["angle_dis_opt"])[:-1], show_text=True, color="k", legend_name="optimized angle")
    plt.xlabel("Rotation in degree")
    plt.ylabel("Percentage")
    plt.title("Angle Distance")
    plt.savefig(save_path / "angle_distance.png", pad_inches=0)
    plt.show()

    plot_cumulated_curve(np.array(data["mse_grid"])[difficult_index], show_text=True, color="skyblue",
                         legend_name="grid mse")
    plot_cumulated_curve(np.array(data["mse_opt"])[difficult_index], show_text=True, color="k", legend_name="optimized mse")
    plt.xlabel("MSE in pixel (512 x 512)")
    plt.ylabel("Percentage")
    plt.title("MSE (AI98 x 11)")
    plt.savefig(save_path / "mse_difficult.png", pad_inches=0)
    plt.show()

    plot_cumulated_curve(np.array(data["angle_dis_grid"])[difficult_index], show_text=True, color="skyblue",
                         legend_name="grid angle")
    plot_cumulated_curve(np.array(data["angle_dis_opt"])[difficult_index], show_text=True, color="k", legend_name="optimized angle")
    plt.xlabel("Rotation in degree")
    plt.ylabel("Percentage")
    plt.title("Angle Distance (AI98 x 11)")
    plt.savefig(save_path / "angle_distance_difficult.png", pad_inches=0)
    plt.show()

    plot_cumulated_curve(np.array(data["mse_grid"])[easy_index], show_text=True, color="skyblue",
                         legend_name="grid mse")
    plot_cumulated_curve(np.array(data["mse_opt"])[easy_index], show_text=True, color="k", legend_name="optimized mse")
    plt.xlabel("MSE in pixel (512 x 512)")
    plt.ylabel("Percentage")
    plt.title("MSE (Other x 20)")
    plt.savefig(save_path / "mse_easy.png", pad_inches=0)
    plt.show()

    plot_cumulated_curve(np.array(data["angle_dis_grid"])[easy_index], show_text=True, color="skyblue",
                         legend_name="grid angle")
    plot_cumulated_curve(np.array(data["angle_dis_opt"])[easy_index], show_text=True, color="k", legend_name="optimized angle")
    plt.xlabel("Rotation in degree")
    plt.ylabel("Percentage")
    plt.title("Angle Distance (Other x 20)")
    plt.savefig(save_path / "angle_distance_easy.png", pad_inches=0)
    plt.show()

    pass
