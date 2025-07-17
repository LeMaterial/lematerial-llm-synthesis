from llm_synthesis.metrics.figure_extraction.generate_scatterplots import (
    generate_synthetic_plots,
)

num_plots = 5
images_path = "./data/figure_output/"
groundtruths_path = "./data/coor_output/"

generate_synthetic_plots(num_plots, images_path, groundtruths_path)
