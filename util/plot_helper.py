import matplotlib as mpl
import matplotlib.font_manager as font_manager


# # ICLR/NeurIPS
# TEXT_WIDTH = 6.00117
# TEXT_HEIGHT = 8.50166
# FOOTNOTE_SIZE = 9
# SCRIPT_SIZE = 8

# AISTATS
TEXT_WIDTH = 6.75133
TEXT_HEIGHT = 9.25182
FOOTNOTE_SIZE = 9
SCRIPT_SIZE = 7

cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
FONT_NAME = cmfont.get_name()



def get_mpl_rcParams(width_percent, height_percent):
    params = {
        'text.usetex': False,
        'font.size': SCRIPT_SIZE,
        'font.family': 'serif',
        'font.serif': FONT_NAME,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.5,
        'axes.titlesize': FOOTNOTE_SIZE,
        'axes.labelsize': SCRIPT_SIZE,
        'axes.unicode_minus': False,
        'axes.formatter.use_mathtext': True,
        'legend.frameon': False,
        'legend.fontsize': SCRIPT_SIZE,
        'legend.handlelength': 1,
        'xtick.major.size': 1.5,
        'ytick.major.size': 1.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    }

    return params, width_percent*TEXT_WIDTH, height_percent*TEXT_HEIGHT
