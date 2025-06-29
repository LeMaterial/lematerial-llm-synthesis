import matplotlib.pyplot as plt
import numpy as np
import random
import os
import string
import csv
import pandas as pd
from scipy.stats import skewnorm
import argparse

num_plots = 5

def generate_random_data(x, curve_type):
    num_points = len(x)

    if curve_type == 'exp_increasing':
        y = np.exp(0.3 * x) + np.random.normal(0, 0.2 * np.exp(0.4 * x), num_points)
    elif curve_type == 'exp_decreasing':
        y = np.exp(-0.3 * x) + np.random.normal(0, 0.2 * np.exp(-0.4 * x), num_points)
    elif curve_type == 'exp_increasing_dec_rate':
        y = np.exp(0.1 * x) + np.random.normal(0, 0.2 * np.exp(0.2 * x), num_points) 
    elif curve_type == 'exp_decreasing_inc_rate':
        y = np.exp(-0.1 * x) + np.random.normal(0, 0.2 * np.exp(-0.2 * x), num_points) 
    elif curve_type == 'linear_steep':
        y = 3 * x + np.random.normal(0, 5, num_points)
    elif curve_type == 'linear_shallow':
        y = 0.5 * x + np.random.normal(0, 2, num_points)

    y += np.random.uniform(-8, 8) 

    return x, y

def random_color(used_colors):
    colors = [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
        'darkblue', 'darkgreen', 'darkred', 'darkcyan', 'darkmagenta', 'gold',
        'navy', 'lime', 'crimson', 'purple', 'orange', 'teal', 'indigo'
    ]
    available_colors = [color for color in colors if color not in used_colors]
    color = random.choice(available_colors)
    used_colors.add(color)
    return color

def random_shape(used_shapes):
    shapes = ['o', 's', 'o', 's', 'o', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    available_shapes = [shape for shape in shapes if shape not in used_shapes]
    shape = random.choice(available_shapes)
    used_shapes.add(shape)
    return shape

def random_string(min_length=5, max_length=15):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def random_x_axis(ltype, filler_axis_labels, x_axis_labels):
    if ltype >= 0.4:
        df = pd.read_csv(filler_axis_labels, header = None)
    else:
        df = pd.read_csv(x_axis_labels, header = None)
    labels = df[0].to_list()
    return random.choice(labels)

def random_y_axis(y_axis_labels):
    df = pd.read_csv(y_axis_labels, header = None)
    labels = df[0].to_list()
    return random.choice(labels)

def random_filler(filler_list):
    df = pd.read_csv(filler_list, header = None)
    fillers = df[0].to_list()
    return random.choice(fillers)

def random_matrix(matrix_list):
    df = pd.read_csv(matrix_list, header = None)
    matrices = df[0].to_list()
    return random.choice(matrices)

def random_legend_variable():
    vars = []
    return random.choice(vars)

def skewed_normal_distribution(num_groups, center=6, skew=-1, size=1):
    if num_groups <= 2:
        center += 2
    return int(np.clip(skewnorm.rvs(a=skew, loc=center, scale=2, size=size), 3, 20)[0])

def place_legend(ax, legend_handles, legend_labels):
    positions = ['upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']
    best_pos = 'upper right'
    min_overlap = float('inf')

    for pos in positions:
        legend = ax.legend(handles=legend_handles, labels=legend_labels, loc=pos, frameon=False)
        bbox = legend.get_window_extent().transformed(ax.transData.inverted())
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1

        overlap = 0
        for scatter in ax.collections:
            offsets = scatter.get_offsets()
            for x, y in offsets:
                if bbox_x0 <= x <= bbox_x1 and bbox_y0 <= y <= bbox_y1:
                    overlap += 1
                    break  # Stop checking this scatter as we already found an overlap
        if overlap < min_overlap:
            min_overlap = overlap
            best_pos = pos
        legend.remove()

    # Place legend in the best position found
    ax.legend(handles=legend_handles, labels=legend_labels, loc=best_pos, frameon=False)

def skewed_marker_size(center=50, skew=4, size=1):
    return np.clip(skewnorm.rvs(a=skew, loc=center, scale=20, size=size), 10, 200)

def plot_random_scatterplot(image_path, csv_path, filler_list, matrix_list, y_axis_labels, x_axis_labels, filler_axis_labels):
    fig, ax = plt.subplots(figsize=(10, 6))

    num_groups = random.randint(1, 5)
    num_points = max(3, min(20, skewed_normal_distribution(num_groups)))
    consistent_x = random.random() < 0.75

    if consistent_x:
        if random.random() < 0.5:
            x = np.sort(np.random.choice(range(0, 20), num_points, replace=False))
        else:
            x = np.sort(np.random.uniform(0, 20, num_points))
    else:
        x = None

    curve_groups = {
        'exponential': ['exp_increasing', 'exp_decreasing', 'exp_increasing_dec_rate', 'exp_decreasing_inc_rate'],
        'linear': ['linear_steep', 'linear_shallow']
    }

    # Choose a primary curve group
    primary_curve_group = random.choice(list(curve_groups.keys()))
    curve_types = curve_groups[primary_curve_group]    
    legend_handles = []
    legend_labels = []
    used_colors = set()
    used_shapes = set()
    data_rows = []
    legend_type = random.random()
    default_mat = random_matrix(matrix_list)
    default_fill = random_filler(filler_list)
    x_label = random_x_axis(legend_type, filler_axis_labels, x_axis_labels)
    y_label = random_y_axis(y_axis_labels)
    marker_size = skewed_marker_size(size=1)

    line_type_prob = random.random()
    if line_type_prob < 0.5:
        line_type = 'no_lines'
    elif line_type_prob < 0.75:
        line_type = 'best_fit'
    else:
        line_type = 'connecting_lines'

    for _ in range(num_groups):
        if not consistent_x:
            if random.random() < 0.5:
                x = np.sort(np.random.choice(range(0, 20), num_points, replace=False))
            else:
                x = np.sort(np.random.uniform(0, 20, num_points))

        curve_type = random.choice(curve_types)
        x, y = generate_random_data(x, curve_type)

        use_color = random.random() < 0.5
        if use_color:
            color = random_color(used_colors)
            shape = random_shape(used_shapes)  # When using colors, shapes can repeat
        else:
            color = 'black'
            shape = random_shape(used_shapes)  # When using black, shapes should be unique
        
        if shape in ['o', 's'] and random.random() < 0.33:
            scatter = ax.scatter(x, y, edgecolor=color, facecolor='none', marker=shape, s=marker_size)
        else:
            scatter = ax.scatter(x, y, c=[color] if use_color else 'black', marker=shape, s=marker_size)

        legend_handles.append(scatter)


        if legend_type < 0.2:
            percentage = f'{round(random.random(), 2)}'
            legend_labels.append(f'{percentage}% {default_fill}/{default_mat}')
        elif legend_type >= 0.2 and legend_type < 0.4:
            percentage = f'{round(random.random(), 2)}'
            legend_labels.append(f'{percentage}% {default_fill}')
        elif legend_type >= 0.4 and legend_type < 0.6:
            legend_labels.append(random_filler(filler_list) + '/' + default_mat)
        elif legend_type >= 0.6 and legend_type < 0.8:
            legend_labels.append(random_matrix(matrix_list))
        if legend_type < 0.8:
            label = legend_labels[-1]
        else:
            label = None

        for i in range(num_points):
            data_rows.append([x[i], y[i], label])

        if line_type == 'best_fit':
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m * x + b, c=color)
        elif line_type == 'connecting_lines':
            ax.plot(x, y, c=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    space_top = random.random() < 0.5
    ylim = ax.get_ylim()
    ylim_range = ylim[1] - ylim[0]
    if space_top:
        ax.set_ylim(ylim[0], ylim[1] + 0.2 * ylim_range)
    else:
        ax.set_ylim(ylim[0] - 0.2 * ylim_range, ylim[1])

    if legend_type <.8:
      place_legend(ax, legend_handles, legend_labels)
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([x_label, y_label])
        writer.writerows(data_rows)

def main():
    parser = argparse.ArgumentParser(description='Generate random scatterplots.')
    parser.add_argument('--resources_path', type=str, required=True, help='Path to the directory containing resource files (CSV files, ie filler_names, matrix_names, etc.).')
    parser.add_argument('--numplots', type=int, required=True, help='Number of scatterplots to generate.')
    parser.add_argument('--images_path', type=str, required=True, help='Path to the directory to save the generated images.')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the directory to save the corresponding CSV labels.')
    parser.add_argument('--start_seed', type=int, required=True, help='First numerical seed used for generating random plots. Ensure this seed does not overlap for training, validation, and/or test data.')


    args = parser.parse_args()
    resources_path = args.resources_path
    num_plots = args.numplots
    images_path = args.images_path
    labels_path = args.labels_path
    start_seed = args.start_seed

    filler_list = os.path.join(resources_path, 'filler_list.csv')
    matrix_list = os.path.join(resources_path, 'matrix_list.csv')
    y_axis_labels = os.path.join(resources_path, 'y_axis_labels.csv')
    x_axis_labels = os.path.join(resources_path, 'x_axis_labels.csv')
    filler_axis_labels = os.path.join(resources_path, 'filler_axis_labels.csv')

    for i in range(start_seed, start_seed + num_plots):
        random.seed(i)
        np.random.seed(i)
        if (i+1)%100 == 0:
            print(f'{i+1}/{num_plots} generated.')
        image_name = f'figure{i+1}.png'
        image_path = os.path.join(images_path, image_name)
        csv_path = os.path.join(labels_path, image_name.replace('png', 'csv'))
        plot_random_scatterplot(image_path, csv_path, filler_list, matrix_list, y_axis_labels, x_axis_labels, filler_axis_labels)


if __name__ == '__main__':
    main()
