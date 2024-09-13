""" This file contains plot utilities """

from PIL import Image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from al_utilities import get_stats, compute_pers_anchoring_relation
import statsmodels.formula.api as smf
import statsmodels.api as sm
from PIL import ImageColor


def latex_plt(matplotlib):
    """ This function updates the matplotlib library to use Latex and changes some default plot parameters

    :param matplotlib: matplotlib instance
    :return: Updated matplotlib instance
    """

    pgf_with_latex = {
        "axes.labelsize": 6,
        "font.size": 6,
        "legend.fontsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.titlesize": 6,
        "pgf.rcfonts": False,
    }
    matplotlib.rcParams.update(pgf_with_latex)

    return matplotlib


def cm2inch(*tupl):
    """ This function converts cm to inches

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


def center_x(cell_lower_left_x, cell_width, word_length, horizontalalignment="left"):
    """ This function centers text along the x-axis

    :param cell_lower_left_x: Lower left x-coordinate
    :param cell_width: Width of cell in which text appears
    :param word_length: Length of plotted word
    :param horizontalalignment: Side of alignment (left vs. center)
    :return: x_center: Centered x-position
    """

    if horizontalalignment == "center":
        x_center = cell_lower_left_x + (cell_width / 2.0)
    else:
        x_center = cell_lower_left_x + (cell_width / 2.0) - (word_length / 2.0)

    return x_center


def center_y(cell_lower_left_y, cell_height, y0, word_height):
    """ This function centers text along the y-axis

    :param cell_lower_left_y: Lower left y-coordinate
    :param cell_height: Height of cell in which text appears
    :param y0: Lower bound of text (sometimes can be lower than cell_lower_left-y (i.e. letter y))
    :param word_height: Height of plotted word
    :return: Centered y-position
    """

    return cell_lower_left_y + ((cell_height / 2.0) - y0) - (word_height / 2.0)


def get_text_coords(f, ax, cell_lower_left_x, cell_lower_left_y, printed_word, fontsize, fontweight='normal'):
    """ This function computes the length and height of a text und consideration of the font size

    :param f: Figure object
    :param ax: Axis object
    :param cell_lower_left_x: Lower left x-coordinate
    :param cell_lower_left_y: Lower left y-coordinate
    :param printed_word: Text of which length is computed
    :param fontsize: Specified font size
    :param fontweight: matplotlib text fontweight input
    :return: word_length: Computed word length
             word_height: Computed word height
             bbox: Text coordinates
    """

    # Print text to lower left cell corner
    t = ax.text(cell_lower_left_x, cell_lower_left_y, printed_word, fontsize=fontsize, fontweight=fontweight)

    # Get text coordinates
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    bbox = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())

    # Compute length and height
    word_length = bbox.x1 - bbox.x0
    word_height = bbox.y1 - bbox.y0

    # Remove printed word
    t.set_visible(False)

    return word_length, word_height, bbox


def plot_centered_text(f, ax, cell_x0, cell_y0, cell_x1, cell_y1, text, fontsize, fontweight='normal', c_type='both',
                       horizontalalignment="left"):
    """ This function plots centered text

    :param f: Figure object
    :param ax: Axis object
    :param cell_x0: Lower left x-coordinate
    :param cell_y0: Lower left y-coordinate
    :param cell_x1: Lower right x-coordinate
    :param cell_y1: Upper left y-coordinate
    :param text: Printed text
    :param fontsize: Current font size
    :param fontweight: matplotlib text fontweight input
    :param c_type: Centering type (y: only y-axis; both: both axes)
    :param horizontalalignment: Side of alignment (left vs. center)
    :return: ax: Axis object
             word_length: Length of printed text
             word_height: Height of printed text
             bbox: Text coordinates
    """

    # Get text coordinates
    word_length, word_height, bbox = get_text_coords(f, ax, cell_x0, cell_y0, text, fontsize)

    # Compute cell width and height
    cell_width = (cell_x1 - cell_x0)
    cell_height = (cell_y1 + cell_y0)

    # Compute centered x position: lower left + half of cell width, then subtract half of word length
    x = center_x(cell_x0, cell_width, word_length, horizontalalignment)

    # Compute centered y position: same as above but additionally correct for word height
    # (because some letters such as y start below y coordinate)
    y = center_y(cell_y0, cell_height, bbox.y0, word_height)

    # Print centered text
    if c_type == 'both':
        ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight)
    elif c_type == 'completely_centered':
        ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight, horizontalalignment=horizontalalignment)
    else:
        ax.text(cell_x0, y, text, fontsize=fontsize, fontweight=fontweight, horizontalalignment='center')

    return ax, word_length, word_height, bbox


def plot_image(f, img_path, cell_x0, cell_x1, cell_y0, ax, text_y_dist, text, text_pos, fontsize,
               zoom=0.2, cell_y1=np.nan, text_col='k'):
    """ This function plots images and corresponding text for the task schematic

    :param f: Figure object
    :param img_path: Path of image
    :param cell_x0: Left x-position of area in which it is plotted centrally
    :param cell_x1: Right x-position of area in which it is plotted centrally
    :param cell_y0: Lower y-position of image
    :param ax: Plot axis
    :param text_y_dist: Y-position distance to image
    :param text: Displayed text
    :param text_pos: Position of printed text (below vs. above)
    :param fontsize: Text font size
    :param zoom: Scale of image
    :param cell_y1: Upper x-position of area in which image is plotted (lower corresponds to cell_y0)
    :param text_col: Text color
    :return ax: Axis object
            bbox: Image coordinates
            ab: Annotation box
    """

    # Open image
    img = Image.open(img_path)

    # Image zoom factor and axis and coordinates
    imagebox = OffsetImage(img, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, (cell_x0, cell_y0), boxcoords="offset points", pad=0, frameon=False)
    ax.add_artist(ab)

    # Get image x coordinate
    cell_width = cell_x1 - cell_x0
    image_x = cell_x0 + (cell_width / 2)

    # Get image y coordinate
    if not np.isnan(cell_y1):
        cell_height = cell_y1 - cell_y0
        image_y = cell_y0 + (cell_height / 2)
    else:
        image_y = cell_y0

    # Remove image and re-plot at correct coordinates
    ab.remove()
    ab = AnnotationBbox(imagebox, (image_x, image_y), boxcoords="offset points", pad=0, frameon=False)
    ax.add_artist(ab)

    # Get image coordinates based on imagebox
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    bbox = imagebox.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())

    # Text x and y coordinates
    if text_pos == 'left_below':
        # Plot text below image
        x = bbox.x0
        y = bbox.y0 - text_y_dist
    elif text_pos == 'centered_below':
        # Plot text centrally below image
        word_length, _, _ = get_text_coords(f, ax, bbox.x0, bbox.y0, text, 6)
        cell_width = bbox.x1 - bbox.x0
        x = center_x(bbox.x0, cell_width, word_length)
        y = bbox.y0 - text_y_dist
    elif text_pos == 'left_top':
        x = bbox.x0
        y = bbox.y1 + text_y_dist
    else:
        # Plot text centrally above image
        word_length, _, _ = get_text_coords(f, ax, bbox.x0, bbox.y0, text, 6)
        cell_width = bbox.x1 - bbox.x0
        x = center_x(bbox.x0, cell_width, word_length)
        y = bbox.y1 + text_y_dist

    # Add text
    ax.text(x, y, text, fontsize=fontsize, color=text_col, zorder=100)

    return ax, bbox, ab


def plot_rec(ax, patches, cell_lower_left_x, width, cell_lower_left_y, height, facecolor="white", edgecolor="k",
             alpha=0.9):
    """ This function plots a rectangle

    :param ax: Axis object
    :param patches: Patches object
    :param cell_lower_left_x: Lower left corner x-coordinate of rectangle
    :param width: Width of rectangle
    :param cell_lower_left_y: Lower left corner y-coordinate of rectangle
    :param height: Height of rectangle
    :param facecolor: Plot face color
    :param edgecolor: Plot edge color
    :param alpha: Current alpha value for transparency
    :return: Axis object
    """

    p = patches.Rectangle(
        (cell_lower_left_x, cell_lower_left_y), width, height, fill=True, facecolor=facecolor, edgecolor=edgecolor,
        alpha=alpha, transform=ax.transAxes, clip_on=False, linewidth=0.5)

    ax.add_patch(p)

    return ax


def label_subplots(f, texts, x_offset=-0.07, y_offset=0.015):
    """ This function labels the subplots

     Obtained from: https://stackoverflow.com/questions/52286497/
     matplotlib-label-subplots-of-different-sizes-the-exact-same-distance-from-corner

    :param f: Figure object
    :param x_offset: Shifts labels on x-axis
    :param y_offset: Shifts labels on y-axis
    :param texts: Subplot labels
    """

    # Get axes
    axes = f.get_axes()

    if isinstance(x_offset, float):
        x_offset = np.repeat(x_offset, len(axes))

    if isinstance(y_offset, float):
        y_offset = np.repeat(y_offset, len(axes))

    # Initialize counter
    axis_counter = 0

    # Cycle over subplots and place labels
    for a, l in zip(axes, texts):
        x = a.get_position().x0
        y = a.get_position().y1
        f.text(x - x_offset[axis_counter], y + y_offset[axis_counter], l, size=12)
        axis_counter += 1


def text_legend(ax, txt, coords=None, loc=None):
    """ This function creates a plot legend that contains text only

    :param ax: Axis object
    :param txt: Legend text
    :param coords: X-Y coordinates
    :param loc: Legend location
    :return: at: Anchored text object
    """

    # Default coordinates
    if coords is None:
        coords = [-0.0, -0.45]

    if loc is None:
        loc = 'lower left'

    # Create legend
    at = AnchoredText(txt, loc=loc, prop=dict(size=6), frameon=True,
                      bbox_to_anchor=coords, bbox_transform=ax.transAxes)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    return at


def plot_arrow(ax, x1, y1, x2, y2, shrink_a=1, shrink_b=1, connectionstyle="arc3,rad=0", arrow_style="<-", color="0.5"):
    """ This function plots arrows for the task schematic

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

    ax.annotate("", xy=(x1, y1), xycoords='data', xytext=(x2, y2), textcoords='data', annotation_clip=False,
                arrowprops=dict(arrowstyle=arrow_style, color=color, shrinkA=shrink_a, shrinkB=shrink_b,
                                patchA=None, patchB=None, connectionstyle=connectionstyle))

    return ax


def plot_header(f, ax, header, patches):
    """ This function plots a row of plots on grid as a header

    :param f: Figure object
    :param ax: Axis object
    :param header: Header text
    :param patches: Patches object
    :return: ax: Axis object
    """

    # Plot background
    hdr_cell_x0, width = 0.0, 1
    hdr_cell_y0, height = 0.3, 1
    ax = plot_rec(ax, patches, hdr_cell_x0, width, hdr_cell_y0, height, alpha=0.5, facecolor="gray",
                  edgecolor="gray")

    # Plot text
    hdr_cell_x1 = hdr_cell_x0 + width
    hdr_cell_y1 = hdr_cell_y0 + height
    plot_centered_text(f, ax, hdr_cell_x0, hdr_cell_y0, hdr_cell_x1, hdr_cell_y1, header, 8)

    # Turn unnecessary axes off
    ax.axis('off')

    return ax


def swarm_boxplot(ax, data, voi, ylabel, exp, alpha=1):
    """ This function combines a boxplot with a swarmplot

    :param ax: Axis object
    :param data: Data set for plotting
    :param voi: Variable of interest
    :param ylabel: Plot y-label
    :param exp: Current experiment
    :param alpha: Current alpha value for transparency
    :return: ax: Axis object
    """

    # Plot colors
    # Futuretodo: If used more broadly, color should be parameterized
    if exp == 1:
        colors = ["#BBE1FA", "#3282B8", "#0F4C75", "#1B262C"]
    else:
        colors = ["#BBE1FA", "#0F4C75", "#1B262C"]

    sns.set_palette(sns.color_palette(colors))

    # Plot seaborn boxplot
    sns.boxplot(x="age_group", y=voi, data=data, hue="age_group",
                notch=False, showfliers=False, linewidth=0.8, width=0.3,
                boxprops=dict(alpha=alpha), ax=ax, showcaps=False, palette=colors, legend=False)

    # Add dots
    sns.stripplot(x="age_group", y=voi, data=data, color='gray', alpha=0.7, size=2)

    # Adjust labels
    if exp == 1:
        plt.xticks(np.arange(4), ['CH', 'AD', 'YA', 'OA'], rotation=0)
    else:
        plt.xticks(np.arange(3), ['CH', 'YA', 'OA'], rotation=0)
    ax.set_xlabel('Age Group')
    plt.ylabel(ylabel)

    return ax


def custom_boxplot_condition(ax, cond_1, cond_2, voi, ylabel, colors, with_lines=True):
    """ This function creates a custom-made boxplot to plot conditions and age groups

        Inspired by
            https: // github.com / mwaskom / seaborn / issues / 979
            https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn#google_vignette

        Potentially relevant
            https://stackoverflow.com/questions/43434020/black-and-white-boxplots-in-seaborn

    :param ax: Axis object
    :param cond_1: First condition for left box
    :param cond_2: Second condition for right box
    :param voi: Variable of interest
    :param ylabel: Plot y-label
    :param colors: Colors for plotting age and condition together
    :param with_lines: Optional single-subject lines emphasizing within-subject differences
    :return: ax: Axis object
    """

    # Extract conditions
    data = {'noPush': cond_1[voi].copy(), 'push': cond_2[voi].copy(),
            'age_group': cond_1['age_group'].copy()}
    df_init = pd.DataFrame(data=data)
    data = df_init.melt(id_vars=['age_group'])

    # Plot seaborn boxplot
    sns.boxplot(x="age_group", y="value", hue='variable', data=data,
                notch=False, showfliers=False, linewidth=0.8, width=0.3,
                ax=ax, showcaps=False, palette='dark:gray')

    # Update colormap
    sns.set_palette(sns.color_palette(colors))

    # Custom facecolors
    counter = 0
    for patch in ax.patches:
        if patch.__class__.__name__ == 'PathPatch':

            r, g, b = ImageColor.getcolor(colors[counter], "RGB")
            r, g, b, = r / 255, g / 255, b / 255
            patch.set_facecolor((r, g, b, 1))  # 0.2
            if (counter % 2) == 0:
                patch.set_facecolor((r, g, b, 1))
            counter += 1

    if with_lines:
        result = pd.merge(cond_1, cond_2, on=["subj_num", "age_group"], suffixes=('_noPush', '_push'))
        age_group = [1, 3, 4]
        for i in range(3):
            pers_ch = result[result['age_group'] == age_group[i]]
            y = [pers_ch[voi + '_noPush'], pers_ch[voi + '_push']]
            x = [i + -0.075, i + 0.075]

            # add jitter
            jitter = np.random.uniform(-0.1, 0.1, len(pers_ch))
            x[0] += jitter
            x[1] += jitter

            ax.plot(x, y, color='gray', alpha=0.8, zorder=0, linewidth=0.1)

    # Remove legend
    ax.get_legend().remove()

    # Adjust labels
    plt.xticks(np.arange(3), ['CH', 'YA', 'OA'], rotation=0)
    ax.set_xlabel('Age Group')
    plt.ylabel(ylabel)

    return ax


def box_line_plot(data, voi, ax, x_label, x_ticklabels, y_label, age_label):
    """ This function plot within-subject differences using a combination of boxplot and lines

    :param data: Data frame
    :param voi: Variable of interest
    :param ax: Plot axis
    :param x_label: x-axis label
    :param x_ticklabels: x-axis tick labels
    :param y_label: y-axis label
    :param age_label: Age group for header
    :return: None
    """

    # Plot seaborn boxplot
    sns.boxplot(x="cond", y=voi, data=data,
                notch=False, showfliers=False, linewidth=0.8, width=0.3,
                boxprops=dict(alpha=1), ax=ax, showcaps=False)

    # Plot seaborn line plot
    ax = sns.lineplot(x="cond", y=voi, data=data, hue='subj_num', markers=False,
                      palette=['gray'] * len(data['subj_num'].unique()), legend=False, linewidth=0.2)

    # Axis labels
    ax.set_xlabel(x_label)
    ax.xaxis.set_ticks([0, 1])
    ax.set_xticklabels(x_ticklabels)
    ax.set_ylabel(y_label)

    # Stats for p-value
    # -----------------

    # Extract the two conditions
    cond_1 = data[data['cond'] == 1].reset_index(drop=True)
    cond_0 = data[data['cond'] == 0].reset_index(drop=True)

    # Compute difference between conditions
    cond_diff = cond_0.copy()
    cond_diff[voi] = cond_diff[voi] - cond_1[voi]

    # Test null hypothesis that the distribution of the differences between conditions
    # is symmetric about zero with the nonparametric Wilcoxon sign rank test
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
    res = stats.wilcoxon(cond_diff[voi], y=None, zero_method='wilcox', correction=False, alternative='two-sided')

    # Plot title incl. p-value
    if res.pvalue < 0.001:
        ax.set_title(age_label + ': p < 0.001')
    else:
        ax.set_title(age_label + ': p = ' + str(round(res.pvalue, 5)))


def plot_validation_results(df_data, df_subj):
    """ This function creates the model-validation plots

         1. Plot trial-by-trial data
         2. Regression analysis no-push condition
         3. Regression analyses push condition

    :param df_data: Simulated data
    :param df_subj: Data frame with experimental data
    """

    # Split data into push and no-push condition
    df_data_no_push = df_data[df_data["cond"] == "main_noPush"]
    df_data_push = df_data[df_data["cond"] == "main_push"]

    # Compute perseveration
    pers = df_data['sim_a_t'] == 0

    # ---------------------------
    # 1. Plot trial-by-trial data
    # ---------------------------

    plot_trial_validation(df_subj, df_data, pers)

    # ----------------------------------------
    # 2. Regression analysis no-push condition
    # ----------------------------------------

    # Plot update as a function of prediction error
    plt.figure()
    axlim = 220
    plt.plot([-axlim, axlim], [-axlim, axlim], color='gray', linestyle='-')
    plt.axhline(y=0.5, color='gray', linestyle='-')
    plt.xlim(-axlim, axlim)
    plt.ylim(-axlim, axlim)
    plt.plot(np.array(df_data_no_push['delta_t']), np.array(df_data_no_push['sim_a_t']), 'o')
    plt.xlabel('Prediction Error')
    plt.ylabel('Update')
    plt.title('Standard Condition')
    sns.despine()

    # Run regression: Update = b_0 + b_1 * PE
    mod = smf.ols(formula='sim_a_t ~ delta_t', data=df_data_no_push)
    res = mod.fit()
    print(res.summary())

    # Plot estimated fixed learning rate
    pred_lr = res.params.Intercept + res.params.delta_t * df_data_no_push['delta_t']
    plt.plot(np.array(df_data_no_push['delta_t']), np.array(pred_lr), color='k', linestyle='-')

    # -------------------------------------
    # 3. Regression analyses push condition
    # -------------------------------------

    # Linear regression analysis without considering push
    # ---------------------------------------------------

    # Plot update as a function of prediction error
    plt.figure()
    plt.plot([-axlim, axlim], [-axlim, axlim], color='gray', linestyle='-')
    plt.axhline(y=0.5, color='gray', linestyle='-')
    plt.plot(np.array(df_data_push['delta_t']), np.array(df_data_push['sim_a_t']), 'o')
    plt.xlim(-axlim, axlim)
    plt.ylim(-axlim, axlim)
    plt.xlabel('Prediction Error')
    plt.ylabel('Update')
    plt.title('Anchoring Condition')
    sns.despine()

    # Run regression: Update = b_0 + b_1 * PE
    mod = smf.ols(formula='sim_a_t ~ delta_t', data=df_data_push)
    res = mod.fit()
    print(res.summary())

    # Plot estimated fixed learning rate
    pred_lr = res.params.Intercept + res.params.delta_t * df_data_push['delta_t']
    plt.plot(np.array(df_data_push['delta_t']), np.array(pred_lr), color='k', linestyle='-')

    # Linear regression analysis with push
    # ------------------------------------

    # Run regression: Update = b_0 + b_1 * PE + b_2 * y_t
    mod = smf.ols(formula='sim_a_t ~ delta_t + sim_y_t', data=df_data_push)
    res = mod.fit()
    print(res.summary())

    # Partial regression plot: update by prediction error
    sm.graphics.plot_partregress(endog='sim_a_t', exog_i='delta_t', exog_others='sim_y_t',
                                 data=df_data_push, obs_labels=False)
    plt.plot([-axlim, axlim], [-axlim, axlim], color='gray', linestyle='-', zorder=0)
    plt.axhline(y=0.5, color='gray', linestyle='-', zorder=0)
    plt.xlim(-axlim, axlim)
    plt.ylim(-axlim, axlim)
    plt.xlabel('e(Prediction Error | other predictors)')
    plt.ylabel('e(Update | other predictors excluding Prediction Error)')
    sns.despine()

    # Partial regression plot: update by push
    sm.graphics.plot_partregress(endog='sim_a_t', exog_i='sim_y_t', exog_others='delta_t',
                                 data=df_data_push, obs_labels=False)
    plt.plot([-axlim, axlim], [-axlim, axlim], color='gray', linestyle='-', zorder=0)
    plt.axhline(y=0.5, color='gray', linestyle='-', zorder=0)
    plt.xlim(-axlim, axlim)
    plt.ylim(-axlim, axlim)
    plt.xlabel('e(Bucket Push | other predictors)')
    plt.ylabel('e(Update | other predictors excluding Bucket Push)')
    sns.despine()


def plot_trial_validation(df_subj, df_data, pers):
    """ This function plots trial-by-trial simulations and indicates when perseveration occured

    :param df_subj: Data frame with experimental data
    :param df_data: Simulated data
    :param pers: Perseveration
    :return: None
    """

    # Size of figure
    fig_height = 8
    fig_width = 15

    # Create figure
    f = plt.figure(figsize=cm2inch(fig_width, fig_height))

    # Turn interactive plotting mode on for debugger
    plt.ion()

    # Create plot grid
    gs_0 = gridspec.GridSpec(1, 4, wspace=0.5, hspace=0.7, top=0.95, left=0.125, right=0.95)

    # Indicate plot range and x-axis
    plot_range = (0, 400)
    x = np.linspace(0, plot_range[1] - plot_range[0] - 1, plot_range[1] - plot_range[0])

    # Create subplot grid
    gs_11 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_0[:, 0:3], hspace=1)

    # Plot block results
    ax_1 = plt.Subplot(f, gs_11[0:2, 0])
    f.add_subplot(ax_1)
    ax_1.plot(x, np.array(df_subj['mu_t'][plot_range[0]:plot_range[1]]), '--',
              x, np.array(df_subj['x_t'][plot_range[0]:plot_range[1]]), '.', color="#090030")
    ax_1.plot(x, np.array(df_data['sim_b_t']), linewidth=2, color="#0c3c78", alpha=1)
    ax_1.plot(x[200:plot_range[1]], np.array(df_data['sim_z_t'][200:plot_range[1]]), '.', color="#04879c", markersize=5)
    ax_1.set_ylabel('Screen unit')
    ax_1.set_xlabel('Trial')
    ax_1.set_ylim(-9, 309)
    ax_1.plot([200, 200], [-9, 309], color='k')
    ax_1.legend(["Helicopter", "Outcome", "Model", "Anchor"],
                loc='center left', framealpha=0.8, bbox_to_anchor=(1, 0.5))

    # Plot persveration
    ax_2 = plt.Subplot(f, gs_11[2, 0])
    f.add_subplot(ax_2)
    ax_2.plot(x, np.array(pers[plot_range[0]:plot_range[1]]), '.', linewidth=2, color="#090030", alpha=1)
    ax_2.set_ylabel('Perseveration')
    ax_2.set_xlabel('Trial')
    ax_2.plot([200, 200], [0, 1], color='k')

    # Delete unnecessary axes
    sns.despine()


def get_catch_trial_results(df_exp, exp):
    """ This function computes the catch-trial descriptive results

    :param df_exp: Data frame
    :param exp: Current experiment
    :return: df_e_t_diff: Estimation-error differences
             desc: Group descriptive results
             stat: Group stats
             zero_stat: Stats test against zero
             effect_size: Effects sizes group stats
             effect_size_zero: Effect sizes zero stats
    """

    # Extract average estimation error grouped by subject ID, group, catch trials, and changepoints
    e_t = df_exp.groupby(['subj_num', 'age_group', 'v_t', 'c_t'])['e_t'].mean().reset_index(drop=False)

    # Drop cp trials
    e_t = e_t[e_t['c_t'] == 0].reset_index(drop=True)

    # Compute difference between catch trials and no-catch trials
    # If catch trials improve performance, we expect negative values
    # E.g., 5 (small EE) - 10 (larger) = 5 - 10 = -5
    e_t_diff = e_t[e_t['v_t'] == 1]['e_t'].values - e_t[e_t['v_t'] == 0]['e_t'].values

    # Extract subject number and age group
    df_e_t_diff = \
        df_exp[['subj_num', 'age_group']].drop_duplicates(subset=['subj_num', 'age_group']).reset_index(drop=True)

    # Add estimation-error difference
    df_e_t_diff['e_t_diff'] = e_t_diff.copy()

    # Inferential stats
    desc, stat, effect_size = get_stats(df_e_t_diff, exp, "e_t_diff")

    # Stats against zero
    _, zero_stat, effect_size_zero = get_stats(df_e_t_diff, exp, "e_t_diff", test=2)

    return df_e_t_diff, desc, stat, zero_stat, effect_size, effect_size_zero


def plot_pers_est_err_reg(pers_noPush, model_exp2, ax):
    """ This function plots the association between perseveration and anchoring for each age group

    :param pers_noPush: Perseveration in standard condition
    :param model_exp2: Model parameters incl. anchoring bias
    :param ax: Axis object
    :return: ax: Axis object
    """

    # Adjust figure colors
    colors = ["#BBE1FA", "#0F4C75", "#1B262C"]
    sns.set_palette(sns.color_palette(colors))

    # Compute association perseveration and anchoring
    res = compute_pers_anchoring_relation(pers_noPush, model_exp2)

    # Plot results
    ax.plot(np.array(pers_noPush[pers_noPush['age_group'] == 1]['pers'].copy()),
            np.array(model_exp2[pers_noPush['age_group'] == 1]['d'].copy()),
            '.', color=colors[0], alpha=1, markersize=5)
    ax.plot(np.array(pers_noPush[pers_noPush['age_group'] == 3]['pers'].copy()),
            np.array(model_exp2[pers_noPush['age_group'] == 3]['d'].copy()),
            '.', color=colors[1], alpha=1, markersize=5)
    ax.plot(np.array(pers_noPush[pers_noPush['age_group'] == 4]['pers'].copy()),
            np.array(model_exp2[pers_noPush['age_group'] == 4]['d'].copy()),
            '.', color=colors[2], alpha=1, markersize=5)
    ax.plot(np.array(pers_noPush[pers_noPush['age_group'] == 1]['pers'].copy()),
            np.array(res.fittedvalues[pers_noPush['age_group'] == 1]),
            '-', label="CH", color=colors[0])
    ax.plot(np.array(pers_noPush[pers_noPush['age_group'] == 3]['pers'].copy()),
            np.array(res.fittedvalues[pers_noPush['age_group'] == 3]),
            '-', label="YA", color=colors[1])
    ax.plot(np.array(pers_noPush[pers_noPush['age_group'] == 4]['pers'].copy()),
            np.array(res.fittedvalues[pers_noPush['age_group'] == 4]),
            '-', label="OA", color=colors[2])
    ax.set_ylabel('Anchoring Bias')
    ax.set_xlabel('Perseveration Probability')
    ax.legend()

    # Delete unnecessary axes
    sns.despine()

    return ax


def plot_sampling_params(a, b, sampled_params, dist):
    """ This function plot the sampling-model parameters for illustration purposes

    :param a: First parameter
    :param b: Second parameter
    :param sampled_params: Samples from the plotted distribution
    :param dist: Distribution type (1: Gamma, 2: Gaussian)
    :return: None
    """

    if dist == 1:

        # Gamma case
        x = np.linspace(stats.gamma.ppf(0.01, a, scale=b),
                        stats.gamma.ppf(0.99, a, scale=b), 100)
        rv = stats.gamma(a, scale=b)
        plt.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
        plt.xlim([x[0], x[-1]])

    else:

        # Gaussian case
        x = np.linspace(stats.norm.ppf(0.01, a, scale=b),
                        stats.norm.ppf(0.99, a, scale=b), 100)
        rv = stats.norm(a, scale=b)
        plt.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
        plt.xlim([x[0], x[-1]])

    # Plot sampled parameters
    plt.hist(sampled_params, density=True)


def plot_sampling_results_row(subplot, f, pers_noPush, pers_push, all_est_errs, df_reg, condition_colors, ylabel_dist,
                              plot_legend=True, title=''):
    """ This function plots the sampling simulations

    :param subplot: Subplot object
    :param f: Figure object
    :param pers_noPush: Perseveration in standard condition
    :param pers_push: Perseveration in anchoring condition
    :param all_est_errs: Estimation-error data frame
    :param df_reg: Anchoring bias data frame
    :param condition_colors: Color list
    :param ylabel_dist: Distance to y label
    :param plot_legend: Legend (optional)
    :param title: Plot title (optional)
    :return: None
    """

    # Plot perseveration probability
    ax_1 = plt.Subplot(f, subplot[0, 0])
    f.add_subplot(ax_1)
    custom_boxplot_condition(ax_1, pers_noPush, pers_push, "value", 'Perseveration\nProbability', condition_colors,
                             with_lines=False)
    ax_1.yaxis.set_label_coords(ylabel_dist, 0.5)  # adjust distance of ylabal

    if plot_legend:

        # Add custom legend
        text_legend(plt.gca(), "Darker colors (left): Standard condition | Lighter colors (right): Anchoring condition",
                    coords=[-0.5, -0.8])

    # Load data from figure 5 for comparison between model and data
    exp2_pers_abs_stand_desc = pd.read_csv(
        '~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/exp2_pers_abs_stand_desc.csv',
        index_col='age_group')
    exp2_est_err_abs_stand_desc = pd.read_csv(
        '~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/exp2_est_err_abs_stand_desc.csv',
        index_col='age_group')
    exp2_est_err_abs_stand_anchor = pd.read_csv(
        '~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/exp2_est_err_abs_anchor_desc.csv',
        index_col='age_group')
    exp2_df_reg_desc = pd.read_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/exp2_df_reg_desc.csv',
                                   index_col='age_group')

    # Plot perseveration
    # X-values: Because both boxplots have width = 0.3, so each 0.15, and we take the middle 0.075 for diamonds
    ms = 3
    blue_diamond = "#0000a5"
    ax_1.plot([-0.075, 0.925, 1.925],
              [exp2_pers_abs_stand_desc['median']['ch'], exp2_pers_abs_stand_desc['median']['ya'],
               exp2_pers_abs_stand_desc['median']['oa']], 'd', color=blue_diamond, zorder=100,
              markersize=ms)
    ax_1.set_ylim(-0.05, 0.7)

    # Plot estimation errors
    ax_2 = plt.Subplot(f, subplot[0, 1])
    f.add_subplot(ax_2)
    est_errs_noPush = all_est_errs[all_est_errs['variable'] == "noPush"].reset_index(drop=True)
    est_errs_push = all_est_errs[all_est_errs['variable'] == "push"].reset_index(drop=True)

    custom_boxplot_condition(ax_2, est_errs_noPush, est_errs_push, "value", 'Estimation Error', condition_colors,
                             with_lines=False)
    ax_2.yaxis.set_label_coords(ylabel_dist, 0.5)
    ax_2.plot([-0.075, 0.925, 1.925],
              [exp2_est_err_abs_stand_desc['median']['ch'], exp2_est_err_abs_stand_desc['median']['ya'],
               exp2_est_err_abs_stand_desc['median']['oa']], 'd', color=blue_diamond, zorder=100,
              markersize=ms)
    ax_2.plot([0.075, 1.075, 2.075],
              [exp2_est_err_abs_stand_anchor['median']['ch'], exp2_est_err_abs_stand_anchor['median']['ya'],
               exp2_est_err_abs_stand_anchor['median']['oa']], 'd', color=blue_diamond, zorder=100,
              markersize=ms)

    ax_2.set_ylim(6, 32)
    ax_2.set_title(title, pad=10)

    # Update colors
    colors = ["#BBE1FA", "#0F4C75", "#1B262C"]
    sns.set_palette(sns.color_palette(colors))

    # Plot anchoring bias
    ax_3 = plt.Subplot(f, subplot[0, 2])
    f.add_subplot(ax_3)
    swarm_boxplot(ax_3, df_reg, 'bucket_bias', ' ', 2)
    ax_3.set_ylabel('Anchoring Bias')
    ax_3.yaxis.set_label_coords(ylabel_dist, 0.5)

    ax_3.plot([0, 1, 2], [exp2_df_reg_desc['median_bb']['ch'], exp2_df_reg_desc['median_bb']['ya'],
                          exp2_df_reg_desc['median_bb']['oa']], 'd', color=blue_diamond, zorder=100, markersize=ms)

    ax_3.set_ylim(-0.05, 0.6)
