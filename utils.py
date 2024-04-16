import os
import matplotlib.pyplot as plt


def vis_airfoil2(source,target,idx,dir_name='output_airfoil',sample_type='ddpm'):
    os.makedirs(dir_name,exist_ok=True)
 
    ## 将source和target放到一张图
    plt.scatter(source[:, 0], source[:, 1], c='red', label='source')  # plot source points in red
    plt.scatter(target[:, 0], target[:, 1], c='blue', label='target')  # plot target points in blue
    plt.legend()  # show legend

    file_path = f'{dir_name}/{sample_type}_{idx}.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    # Clear the plot cache
    plt.clf()

def vis_airfoil3(source,target_pred,target,idx,dir_name='output_airfoil',sample_type='ddpm'):
    os.makedirs(dir_name,exist_ok=True)
 
    ## 将source和target放到一张图
    plt.scatter(source[:, 0], source[:, 1], c='red', label='source')  # plot source points in red
    plt.scatter(target_pred[:, 0], target_pred[:, 1], c='green', label='target_pred')  #  plot target pred points in blue
    plt.scatter(target[:, 0], target[:, 1], c='blue', label='target')  # plot target points in blue
    plt.legend()  # show legend

    file_path = f'{dir_name}/{sample_type}_{idx}.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    # Clear the plot cache
    plt.clf()