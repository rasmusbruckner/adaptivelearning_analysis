# This script contains plot utilities

from PIL import Image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from al_utilities import get_mean_voi, get_stats


def latex_plt(matplotlib):
    """ This function updates the matplotlib library to use Latex and changes some default plot parameters

    :param matplotlib: matplotlib instance
    :return: updated matplotlib instance
    """

    pgf_with_latex = {
        # "pgf.texsystem": "pdflatex",
        # "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": [],
        "axes.labelsize": 6,
        "font.size": 6,
        "legend.fontsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.titlesize": 6,
        "pgf.rcfonts": False,
        "text.latex.unicode": True,
        "pgf.preamble": [
             r"\usepackage[utf8x]{inputenc}",
             r"\usepackage[T1]{fontenc}",
             r"\usepackage{cmbright}",
             ]
    }
    matplotlib.rcParams.update(pgf_with_latex)

    return matplotlib


def cm2inch(*tupl):
    """ This function convertes cm to inches

    Obtained from: https://stackoverflow.com/questions/14708695/
    specify-figure-size-in-centimeter-in-matplotlib/22787457

    :param tupl: Size of plot in cm
    :return: Converted image size in inches
    """

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def center_x(cell_lower_left_x, cell_width, word_length):
    """ This function centers text along the x-axis

    :param cell_lower_left_x: Lower left x-coordinate
    :param cell_width: Width of cell in which text appears
    :param word_length: Length of plotted word
    :return: Centered x-position
    """

    return cell_lower_left_x + (cell_width / 2.0) - (word_length / 2.0)


def center_y(cell_lower_left_y, cell_height, y0, word_height):
    """ This function centers text along the y-axis

    :param cell_lower_left_y: Lower left y-coordinate
    :param cell_height: Height of cell in which text appears
    :param y0: Lower bound of text (sometimes can be lower than cell_lower_left-y (i.e. letter y))
    :param word_height: Height of plotted word
    :return: Centered y-position
    """

    return cell_lower_left_y + ((cell_height / 2.0) - y0) - (word_height / 2.0)


def get_text_coords(f, ax, cell_lower_left_x, cell_lower_left_y, printed_word, fontsize):
    """ This function computes the length and height of a text und consideration of the font size

    :param f: Figure object
    :param ax: Axis object
    :param cell_lower_left_x: Lower left x-coordinate
    :param cell_lower_left_y: Lower left y-coordinate
    :param printed_word: Text of which length is computed
    :param fontsize: Specified font size
    :return: word_length, word_height, bbox: Computed word length and height and text coordinates
    """

    # Print text to lower left cell corner
    t = ax.text(cell_lower_left_x, cell_lower_left_y, printed_word, fontsize=fontsize)

    # Get text coordinates
    f.canvas.draw()
    bbox = t.get_window_extent().inverse_transformed(ax.transData)
    word_length = bbox.x1 - bbox.x0
    word_height = bbox.y1 - bbox.y0

    # Remove printed word
    t.set_visible(False)

    return word_length, word_height, bbox


def plot_centered_text(f, ax, cell_x0, cell_y0, cell_x1, cell_y1,
                       text, fontsize, fontweight='normal', c_type='both'):
    """ This function plots centered text

    :param f: Figure object
    :param ax: Axis object
    :param cell_x0: Lower left x-coordinate
    :param cell_y0: Lower left y-coordinate
    :param cell_x1: Lower right x-coordinate
    :param cell_y1: Lower upper left y-coordinate
    :param text: Printed text
    :param fontsize: Current font size
    :param fontweight: Current font size
    :param c_type: Centering type (y: only y axis; both: both axes)
    :return: ax, word_length, word_height, bbox: Axis object, length and height of printed text, text coordinates
    """

    # Get text coordinates
    word_length, word_height, bbox = get_text_coords(f, ax, cell_x0, cell_y0, text, fontsize)

    # Compute cell width and height
    cell_width = (cell_x1 - cell_x0)
    cell_height = (cell_y1 + cell_y0)

    # Compute centered x position: lower left + half of cell width, then subtract half of word length
    x = center_x(cell_x0, cell_width, word_length)

    # Compute centered y position: same as above but additionally correct for word height
    # (because some letters such as y start below y coordinate)
    y = center_y(cell_y0, cell_height, bbox.y0, word_height)

    # Print centered text
    if c_type == 'both':
        ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight)
    else:
        ax.text(cell_x0, y, text, fontsize=fontsize, fontweight=fontweight)

    return ax, word_length, word_height, bbox


def plot_image(f, img_path, cell_x0, cell_x1, cell_y0, ax, text_y_dist, text, text_pos, fontsize,
               zoom=0.2, cell_y1=np.nan):
    """ This function plots images and corresponding text for the task schematic

    :param f: Figure object
    :param img_path: Path of image
    :param cell_x0: Left x-position of area in which it is plotted centrally
    :param cell_x1: Rigth x-position of area in which it is plotted centrally
    :param cell_y0: Lower y-position of image -- if cell_y1 = nan
    :param ax: Plot axis
    :param text_y_dist: y-position distance to image
    :param text: Displayed text
    :param text_pos: Position of printed text (below vs. above)
    :param fontsize: Text font size
    :param zoom: Scale of image
    :param cell_y1: Upper x-position of area in which image is plotted (lower corresponds to cell_y0)
    :return ax, bbox: Axis object, image coordinates
    """

    # Open image
    img = Image.open(img_path)

    # Image zoom factor and axis and coordinates
    imagebox = OffsetImage(img, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, (cell_x0, cell_y0), xybox=None,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0, frameon=False)
    ax.add_artist(ab)

    # Get cell width
    cell_width = cell_x1 - cell_x0
    image_x = cell_x0 + (cell_width/2)

    if not np.isnan(cell_y1):
        cell_height = cell_y1 - cell_y0
        image_y = cell_y0 + (cell_height / 2)
    else:
        image_y = cell_y0

    # Remove image and re-plot at correct coordinates
    ab.remove()
    ab = AnnotationBbox(imagebox, (image_x, image_y), xybox=None,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0, frameon=False)
    ax.add_artist(ab)

    # Get image coordinates
    f.canvas.draw()
    renderer = f.canvas.renderer
    # bbox = imagebox.get_window_extent(renderer).inverse_transformed(ax.transAxes)
    bbox = imagebox.get_window_extent(renderer).inverse_transformed(ax.transData)

    if text_pos == 'left_below':
        # Plot text below image
        x = bbox.x0
        y = bbox.y0 - text_y_dist
    elif text_pos == 'centered_below':
        # Plot text centrally above image
        word_length, _, _ = get_text_coords(f, ax, bbox.x0, bbox.y0, text, 6)
        cell_width = bbox.x1 - bbox.x0
        x = center_x(bbox.x0, cell_width, word_length)
        y = bbox.y0 - text_y_dist
    else:
        # Plot text centrally above image
        word_length, _, _ = get_text_coords(f, ax, bbox.x0, bbox.y0, text, 6)
        cell_width = bbox.x1 - bbox.x0
        x = center_x(bbox.x0, cell_width, word_length)
        y = bbox.y1 + text_y_dist

    ax.text(x, y, text, fontsize=fontsize, color='k')

    return ax, bbox, ab


def plot_arrow(ax, x1, y1, x2, y2, shrink_a=1, shrink_b=1, connectionstyle="arc3,rad=0", arrow_style="<-", color="0.5"):
    """ This function plot arrows for the task schematic

    :param ax: Axis object
    :param x1: x-position of starting point
    :param y1: y-position of starting point
    :param x2: x-position of end point
    :param y2: y-position of end point
    :param shrink_a: Degree with which arrow is decreasing at starting point
    :param shrink_b: Degree with which arrow is decreasing at end point
    :param connectionstyle: Style of connection line
    :param arrow_style: Style of arrow
    :param color: arrow color
    :return ax: Axis object
    """

    ax.annotate("", xy=(x1, y1), xycoords='data', xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle=arrow_style, color=color, shrinkA=shrink_a, shrinkB=shrink_b,
                                patchA=None, patchB=None, connectionstyle=connectionstyle))

    return ax


def plot_rec(ax, patches, cell_lower_left_x, width, cell_lower_left_y, height):
    """ This function plots a rectangle

    :param ax: Axis object
    :param patches: Patches object
    :param cell_lower_left_x: Lower left corner x coordinate of rectangle
    :param width: Width of rectangle
    :param cell_lower_left_y: Lower left corner y coordinate of rectangle
    :param height: Height of rectangle
    :return: Axis object
    """

    p = patches.Rectangle(
        (cell_lower_left_x, cell_lower_left_y), width, height,
        fill=False, transform=ax.transAxes, clip_on=False, linewidth=0.5)

    ax.add_patch(p)

    return ax


def label_subplots(f, texts, x_offset=-0.07, y_offset=0.015):
    """ This function labels the subplots

     Obtained from: https://stackoverflow.com/questions/52286497/
     matplotlib-label-subplots-of-different-sizes-the-exact-same-distance-from-corner

    :param f: Figure handle
    :param x_offset: Shifts labels on x-axis
    :param y_offset: Shifts labels on y-axis
    :param texts: Subplot labels
    """

    # Get axes
    axes = f.get_axes()

    # Cycle over subplots and place labels
    for a, l in zip(axes, texts):
        x = a.get_position().x0
        y = a.get_position().y1
        f.text(x - x_offset, y + y_offset, l, size=12)


def custom_boxplot(ax, data, voi, alpha=1):
    """ This function creates a custom-made boxplot

    :param ax: Current plot axis
    :param data: Data set that for plotting
    :param voi: Variable of interest
    :param alpha: Current alpha value for transparency
    :return: ax: Current plot axis
    """

    # Extract variable of interest and age groups
    x = np.array(data["age_group"])
    y = np.array(data[voi])

    # Plot standard boxplot
    sns.boxplot(x=x, y=y,
                notch=False, showfliers=False, linewidth=0.8, width=0.3,
                boxprops=dict(alpha=alpha), ax=ax)

    # Adjust boxplot
    # --------------

    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the same colour as above
    a = len(ax.artists) * 5
    for j in range(a):

        if j == 2 or j == 3 or j == 7 or j == 8 or j == 12 or j == 13 or j == 17 or j == 18:
            col = 'white'
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)

    return ax


def swarm_boxplot(ax, data, voi, ylabel, exp, alpha=1):
    """ This function combines a boxplot with a swarmplot

    :param ax: Current plot axis
    :param data: Data set that for plotting
    :param voi: Variable of interest
    :param ylabel: Plot ylabel
    :param exp: Current experiment
    :param alpha: Current alpha value for transparency
    :return: ax: Current plot axis
    """

    # Plot boxplot and swarmplot
    ax = custom_boxplot(ax, data, voi, alpha)
    sns.swarmplot(x="age_group", y=voi, data=data, color='gray', alpha=0.7, size=2)

    # Adjust labels
    if exp == 1:
        plt.xticks(np.arange(4), ['CH', 'AD', 'YA', 'OA'], rotation=0)
    else:
        plt.xticks(np.arange(3), ['CH', 'YA', 'OA'], rotation=0)
    ax.set_xlabel('Age group')
    plt.ylabel(ylabel)

    return ax


def get_cond_diff(cond_1, cond_2, voi):
    """ This function computes the differences in perseveration and estimation errors between
        two task conditions in the follow-up experiment

    :param cond_1: First condition of interest
    :param cond_2: Second condition of interest
    :param voi: Variable of interes
    :return: desc, stat, zero_stat: descriptive, inferential statistics and inferential statistics against zero
    """

    # Identify variable of interest
    if voi == 1:
        voi_name = 'e_t'
        print_name = 'Estimation error'
    elif voi == 2:
        voi_name = 'pers'
        print_name = 'Perseveration'
    else:
        voi_name = 'motor_pers'
        print_name = 'Motor perseveration'

    # Compute mean of variable of interest
    voi_cond_1 = get_mean_voi(cond_1, voi)
    voi_cond_2 = get_mean_voi(cond_2, voi)

    # Compute difference between conditions
    cond_diff = voi_cond_2.copy()
    cond_diff[voi_name] = cond_diff[voi_name] - voi_cond_1[voi_name]

    print('\n\n' + print_name + ' difference\n')
    median_diff, q1_diff, q3_diff, p_values_diff, stat_diff = get_stats(cond_diff, 2, voi_name)
    stat = pd.DataFrame()
    stat['p'] = p_values_diff
    stat['stat'] = stat_diff
    stat.index.name = 'test'
    stat = stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'}, axis='index')

    desc = pd.DataFrame()
    desc['median'] = round(median_diff, 3)
    desc['q1'] = round(q1_diff, 3)
    desc['q3'] = round(q3_diff, 3)
    desc.index.name = 'age_group'
    desc = desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

    print('\n\n' + print_name + ' difference test against zero\n')
    _, _, _, p_values_diff, stat_diff = get_stats(cond_diff, 2, voi_name, test=2)
    zero_stat = pd.DataFrame()
    zero_stat['p'] = p_values_diff
    zero_stat['stat'] = stat_diff
    zero_stat.index.name = 'age_group'
    zero_stat = zero_stat.rename({0: 'ch', 1: 'ya', 2: 'oa'}, axis='index')

    return cond_diff, desc, stat, zero_stat


def text_legend(ax, txt):
    """ This function creates a plot legend that contains text only

    :param ax: Current plot axis
    :param txt: Legend text
    :return: at: anchored text object
    """

    at = AnchoredText(txt, loc='lower left', prop=dict(size=6), frameon=True,
                      bbox_to_anchor=(-0.0, -0.45), bbox_transform=ax.transAxes)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    return at
