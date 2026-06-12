import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random


def visualize_and_save_data(base_folder='archive', num_rows=3, num_cols=3, save_name='data_preview.png'):
    """
    作用:
        从训练集中随机抽取若干图片，拼成预览图并保存到本地。
    输入:
        base_folder: 数据根目录，默认 archive。
        num_rows: 预览网格行数。
        num_cols: 预览网格列数。
        save_name: 输出预览图片文件名。
    输出:
        无返回值；成功时保存 data_preview.png 或用户指定文件。
    """
    # 1. 定义文件路径
    train_csv_path = os.path.join(base_folder, 'train.csv')
    train_images_dir = os.path.join(base_folder, 'train_images')

    # 检查路径是否存在
    if not os.path.exists(train_csv_path) or not os.path.exists(train_images_dir):
        print(f"错误：找不到路径。请确认 '{base_folder}' 文件夹在当前目录下。")
        return

    # 2. 读取 CSV 文件
    print("正在读取 CSV 文件...")
    try:
        df = pd.read_csv(train_csv_path)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 3. 随机抽取样本 (num_rows * num_cols 张图片)
    total_images = num_rows * num_cols
    # 如果数据量不足，就取全部
    if len(df) > total_images:
        sample_df = df.sample(total_images).reset_index(drop=True)
    else:
        sample_df = df.reset_index(drop=True)

    # 4. 创建画布
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    # 将 axes 展平为一维数组，方便遍历 (如果是 1x1 的情况特殊处理)
    if total_images > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    print("正在生成图片预览...")

    for i, ax in enumerate(axes):
        if i < len(sample_df):
            row = sample_df.iloc[i]
            img_name = row['image']
            species = row['species']
            individual_id = row['individual_id']

            img_path = os.path.join(train_images_dir, img_name)

            # 读取并显示图片
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)

                # 设置标题 (物种 和 ID)
                title_text = f"Img: {img_name}\nSpecies: {species}\nID: {individual_id}"
                ax.set_title(title_text, fontsize=10, color='blue')
            else:
                ax.text(0.5, 0.5, 'Image Not Found', ha='center', va='center')
                ax.set_title(img_name)

        # 隐藏坐标轴刻度
        ax.axis('off')

    # 调整布局，避免标题重叠
    plt.tight_layout()

    # 5. 保存图片到本地
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"成功！预览图片已保存为: {os.path.abspath(save_name)}")

    # 如果你在 Jupyter Notebook 中运行，取消下面这行的注释可以显示图片
    # plt.show()


if __name__ == "__main__":
    # 运行主函数
    visualize_and_save_data()
