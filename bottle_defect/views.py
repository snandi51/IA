from random import random

from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
import tensorflow.keras.backend as K
from tensorflow import keras
from matplotlib import pyplot
import matplotlib
matplotlib.use('Agg')
import os
import json
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import cv2
import torch
import torchvision

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage.metrics import structural_similarity

import glob
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import iqr
from PIL import Image


# Create your views here.
def login_user(request):
    if request.method == 'POST':
        request.session['username'] = request.POST.get('username')
        password = request.POST.get('password')
        username = request.session.get('username')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            context = {}
            return render(request, 'select.html', context)
        else:
            messages.success(request, 'Invalid Username or Password')
            return render(request, 'login.html')
    if request.user.is_authenticated:
        context = {'authorized': True}
        return render(request, 'select.html', context)
    else:
        return render(request, 'login.html')


def select(request):
    if request.method == 'POST':
        bot = request.POST.get('bot')
        cap = request.POST.get('cap')
        tra = request.POST.get('tra')
        scr = request.POST.get('scr')

        context = {
            'bot': bot,
            'cap': cap,
            'tra': tra,
            'scr': scr,
        }
        return render(request, 'introduction.html', context)
    return render(request, 'select.html')


@login_required()
def introduction(request):
    if request.method == 'POST':
        assessment_portfolio = request.POST.get('assessment_portfolio')
        request.session['assessment_portfolio'] = assessment_portfolio
        # assessment_portfolio = request.session.get('assessment_portfolio')
        if assessment_portfolio == '1':
            portfolio_screen = '1'
        else:
            portfolio_screen = '2'
        context = {
            'assessment_portfolio': request.session['assessment_portfolio'],
            'portfolio_screen': portfolio_screen,
        }
        return render(request, 'upload_file.html', context)
    return render(request, 'introduction.html')

import copy
from pathlib import Path


def get_img_dict(directory: Path):
    tensor_dict = {}

    for dir_ in directory.iterdir():
        dir_name = dir_.parts[-1]

        # print(dir_, dir_name)
        curr_imgs = glob.glob(f"{dir_}/*.png")
        curr_np_arr = np.array([
            np.array(Image.open(fname).resize((256, 256)))
            for fname in curr_imgs
        ])
        curr_np_arr = curr_np_arr.astype(np.float32)
        curr_np_arr /= 255.
        img_name = curr_imgs
        print(dir_name, curr_np_arr.shape)
        tensor_dict[dir_name] = curr_np_arr

    return tensor_dict


def create_in_pred_res_dict(model, images_dict):
    dict_tuples = {}
    for k, imgs in images_dict.items():
        curr_pred = model.predict(imgs)
        curr_resmap = calculate_resmaps(imgs, curr_pred, method="l2")[1]
        dict_tuples[k] = (imgs, curr_pred, curr_resmap)
    return dict_tuples


def calculate_resmaps(imgs_input, imgs_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # if RGB, transform to grayscale and reduce tensor dimension to 3
    if imgs_input.ndim == 4 and imgs_input.shape[-1] == 3:
        imgs_input_gray = tf.image.rgb_to_grayscale(imgs_input).numpy()[:, :, :, 0]
        imgs_pred_gray = tf.image.rgb_to_grayscale(imgs_pred).numpy()[:, :, :, 0]
    else:
        imgs_input_gray = imgs_input
        imgs_pred_gray = imgs_pred

    # calculate remaps
    if method == "l2":
        scores, resmaps = resmaps_l2(imgs_input_gray, imgs_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(imgs_input_gray, imgs_pred_gray)
    if dtype == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return scores, resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps


def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
        # resmap = np.expand_dims(resmap, axis=-1)
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return scores, resmaps


model_path = f"{os.getcwd()}\\bottle/model/mvtecCAE_b8_e6.hdf5"


def load_model_HDF5(model_path):
    """
    Loads model (HDF5 format), training setup and training history.
    """

    # load parameters
    info = get_model_info(model_path)
    loss = info["model"]["loss"]
    dynamic_range = info["preprocessing"]["dynamic_range"]

    # load autoencoder
    if loss == "mssim":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "loss": mssim_loss(dynamic_range),
                "mssim": mssim_metric(dynamic_range),
            },
            compile=True,
        )

    elif loss == "ssim":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "loss": ssim_loss(dynamic_range),
                "ssim": ssim_metric(dynamic_range),
            },
            compile=True,
        )

    else:
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "l2_loss": l2_loss,
                "ssim": ssim_loss(dynamic_range),
                "mssim": mssim_metric(dynamic_range),
            },
            compile=True,
        )

    # load training history
    dir_name = os.path.dirname(model_path)
    history = pd.read_csv(os.path.join(dir_name, "history.csv"))

    return model, info, history


def l2_loss(imgs_true, imgs_pred):
    # return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)
    return tf.nn.l2_loss(imgs_true - imgs_pred)


# losses.py
def ssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):

        # return (1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)) / 2

        return 1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)

        # return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, dynamic_range))

    return loss


# metrics.py
def ssim_metric(dynamic_range):
    def ssim(imgs_true, imgs_pred):
        return K.mean(tf.image.ssim(imgs_true, imgs_pred, dynamic_range), axis=-1)

    return ssim


def mssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):

        return 1 - tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)

        # return 1 - tf.reduce_mean(
        #     tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)
        # )

    return loss


def mssim_metric(dynamic_range):
    def mssim(imgs_true, imgs_pred):
        return K.mean(
            tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range), axis=-1
        )

    return mssim


# custom function to load model
def get_model_info(model_path):
    dir_name = os.path.dirname(model_path)
    print(os.getcwd())
    with open(f"{os.getcwd()}/bottle/model/info.json", "r") as read_file:
        info = json.load(read_file)
    return info


bottle_model, info, _ = load_model_HDF5(model_path)
bottle_model.summary()


# returned from create_in_pred_res_dict
def create_images_batch_from_dict(input_gen_res_data_dict):
    all_merged_inp = []
    all_merged_gen = []
    all_merged_res = []
    all_merged_labels = []

    for lbl, np_arr in input_gen_res_data_dict.items():
        input_img, gen_img, res_map = np_arr
        all_merged_inp.append(input_img)
        all_merged_gen.append(gen_img)
        all_merged_res.append(res_map)

        all_merged_labels.extend([lbl] * input_img.shape[0])

    all_merged_inp = np.vstack(all_merged_inp)
    all_merged_gen = np.vstack(all_merged_gen)
    all_merged_res = np.vstack(all_merged_res)

    all_merged_data = (all_merged_inp, all_merged_gen, all_merged_res)
    print(all_merged_inp.shape, all_merged_gen.shape, all_merged_res.shape, len(all_merged_labels))

    return all_merged_data, all_merged_labels


def get_img_dict_input(directory: Path):

    tensor_dict = {}

    dir_name=directory.name

    fname=(dir_name.split("."))[0]

    curr_np_arr = np.array([

        np.array(Image.open(directory).resize((256, 256)))

    ])

    curr_np_arr = curr_np_arr.astype(np.float32)

    curr_np_arr /= 255.

    tensor_dict[fname] = curr_np_arr

    print(dir_name, curr_np_arr.shape)

    return tensor_dict


# convert from NHWC to NCHW, for pyTorch
def plot_img_grid(batch_img_tensor, fig_size=(54, 6)):
    # batch_img_tensor: numpy array in NHWC format
    torch_tensor = torch.from_numpy(batch_img_tensor)
    # print(torch_tensor.shape)

    torch_tensor = torch_tensor.permute(0, 3, 1, 2) # from NHWC to NCHW
    # print(torch_tensor.shape)
    grid_img = torchvision.utils.make_grid(torch_tensor)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(grid_img.permute(1, 2, 0), interpolation='nearest')
    plt.tight_layout()


from scipy import stats as st


def bottles_prediction(bottle_model,
                       data_samples: np.array,
                       data_samples_input: np.array,
                       data_labels_input: list,
                       data_labels: list,
                       bottles_data_dict: dict,
                       bottles_data_dict_input: dict,
                       Z_FACTOR: float = 1.96, # 95 %CI, use 2.57 for 99% CI
                       ):

    #limit = min(limit, data_samples_input.shape[0])
    #print(limit)
    HISTOGRAM_BINS = 25
    VALS_FILTER = 0.07
    Z_FACTOR = 1.96
    COLOR_MAP = "Spectral" # plasma, inferno, cviridis
    MODEL = bottle_model
    curr_imgs = data_samples_input
    if len(data_labels_input) == 0:
        data_labels_input = ["TEST"] * curr_imgs.shape[0]
    labels = data_labels_input
    assert curr_imgs.shape[0] == len(labels), "data points and labels should match"

    generated_images = MODEL.predict(curr_imgs)
    _, residual_maps = calculate_resmaps(curr_imgs, generated_images, method="l2")

    #curr_imgs, generated_images, residual_maps = curr_imgs[:limit], generated_images[:limit], residual_maps[:limit]

    idx = 0
    temp_tbl = {"img":[], # input/good

                "vals_count":[],
                "mean":[],
                "std":[],
                "IQR":[],
                "min":[],
                "25":[],
                "50":[],
                "75":[],
                "max":[],

                }

    final_stat_dict_bottle = {
        # "random_samples_idx": [],
        "lower_bound_iqr":[],
        "upper_bound_iqr": [],
        "input_img_iqr":[],
        "lower_bound_range":[],
        "upper_bound_range": [],
        "input_img_range":[],
        "lift_iqr":[],
        "actual_lbl": [],
        "prediction": [],
        "prediction_result": [],
        "lift_range": []
    }
    plots = []

    # for COLOR_MAP in ["Greys", "plasma", "Spectral", "seismic", "CMRmap"]:
    for input_img, gen_img, resmap, lbl in zip(curr_imgs, generated_images, residual_maps, labels):
        idx += 1
        curr_temp_tbl = copy.deepcopy(temp_tbl)

        fig1, ax = plt.subplots(1, 2, figsize=(9, 5))
        ax[0].imshow(input_img)
        ax[0].title.set_text("input("+lbl+")")

        # ax[1].imshow(gen_img)
        # ax[1].title.set_text("generated")

        ax[1].imshow(resmap, cmap="plasma")
        ax[1].title.set_text("resmap")
        # fig1.show()
        fig1.savefig("media/fig1/my_resmap_"+lbl)

        fig2,ax2 = plt.subplots(1, 2, figsize=(9, 6))

        in_resmap = resmap.flatten()
        in_vals = in_resmap[in_resmap > VALS_FILTER]

        #in_vector = np.histogram(in_vals, bins=HISTOGRAM_BINS, range=(0, 1))
        curr_temp_tbl["img"].append("input")
        curr_temp_tbl["vals_count"].append(len(in_vals))
        curr_temp_tbl["mean"].append(in_vals.mean())
        curr_temp_tbl["std"].append(in_vals.std(ddof=1))
        curr_temp_tbl["IQR"].append(iqr(in_vals))
        curr_temp_tbl["min"].append(in_vals.min())
        curr_temp_tbl["25"].append(np.quantile(in_vals, 0.25))
        curr_temp_tbl["50"].append(np.quantile(in_vals, 0.5))
        curr_temp_tbl["75"].append(np.quantile(in_vals, 0.75))
        curr_temp_tbl["max"].append(in_vals.max())

        #ax2[0][0].imshow(resmap, cmap=COLOR_MAP)
        #pyplot.title.set_text("in_res")
        fitted_data, fitted_lambda = st.boxcox(in_vals)
        ax2[0].hist(abs(fitted_data), bins=HISTOGRAM_BINS)
        #pyplot.hist(in_vals, bins=HISTOGRAM_BINS,alpha=0.1, density = True,histtype ='bar', label="in_res")

        five_random_indexes = [0, 1, 2, 3, 5]
        good_examples = bottles_data_dict["good"][0][five_random_indexes]
        good_examples_gen = MODEL.predict(good_examples)
        _, good_examples_resmaps = calculate_resmaps(good_examples, good_examples_gen, method="l2")

        for idx, g_resmap in zip(range(1, 6), good_examples_resmaps):
            vals = g_resmap.flatten()
            vals = vals[vals > VALS_FILTER]

            curr_temp_tbl["img"].append(f"good_{five_random_indexes[idx - 1]}")
            curr_temp_tbl["vals_count"].append(len(vals))
            curr_temp_tbl["mean"].append(vals.mean())
            curr_temp_tbl["std"].append(vals.std(ddof=1))
            curr_temp_tbl["IQR"].append(iqr(vals))
            curr_temp_tbl["min"].append(vals.min())
            curr_temp_tbl["25"].append(np.quantile(vals, 0.25))
            curr_temp_tbl["50"].append(np.quantile(vals, 0.5))
            curr_temp_tbl["75"].append(np.quantile(vals, 0.75))
            curr_temp_tbl["max"].append(vals.max())

            #ax2[0][idx].title.set_text(f"good_res_{idx}")
            #ax2[0][idx].imshow(g_resmap, cmap=COLOR_MAP)
            fitted_data_good, fitted_lambda_good = st.boxcox(vals)
            pyplot.hist(abs(fitted_data_good), bins=HISTOGRAM_BINS,histtype ='step', label=f"good_res_{idx}")

        #plots.append((fig2))
        #plt.show()
        # pyplot.legend(loc='upper right')
        pyplot.show()

        fig2.savefig("media/fig2/my_graph_"+lbl)

        # AVERAGE OF LAST 5
        curr_temp_tbl["img"].append("avg")
        curr_temp_tbl["vals_count"].append(np.mean(curr_temp_tbl["vals_count"][-5:]))
        curr_temp_tbl["mean"].append(np.mean(curr_temp_tbl["mean"][-5:]))
        curr_temp_tbl["std"].append(np.mean(curr_temp_tbl["std"][-5:]))
        curr_temp_tbl["IQR"].append(np.mean(curr_temp_tbl["IQR"][-5:]))
        curr_temp_tbl["min"].append(np.mean(curr_temp_tbl["min"][-5:]))
        curr_temp_tbl["25"].append(np.mean(curr_temp_tbl["25"][-5:]))
        curr_temp_tbl["50"].append(np.mean(curr_temp_tbl["50"][-5:]))
        curr_temp_tbl["75"].append(np.mean(curr_temp_tbl["75"][-5:]))
        curr_temp_tbl["max"].append(np.mean(curr_temp_tbl["max"][-5:]))

        comparison_df = pd.DataFrame(curr_temp_tbl)
        comparison_df["range"] = comparison_df["max"] - comparison_df["min"]
        comparison_df["marginal_error"] = (comparison_df["IQR"]*1.96)/comparison_df["vals_count"]
        comparison_df["lower_bound"] = comparison_df["IQR"]-comparison_df["marginal_error"]
        comparison_df["Upper_bound"] = comparison_df["IQR"]+comparison_df["marginal_error"]
        #display(comparison_df)
        print("-----------------------------------------------------")
        #display(comparison_df.loc[[0, 6], ["img", "IQR", "range"]])

        good_stats = comparison_df.query("img == 'avg'")[["IQR"]]

        iqr_mean = good_stats.loc[:, "IQR"].values[0]
        iqr_std = np.std(curr_temp_tbl["IQR"][-6:-1])
        marginal_err = Z_FACTOR * iqr_std / 4
        iqr_lower_bound, iqr_upper_bound = iqr_mean - marginal_err, iqr_mean + marginal_err
        #print(f"99% CI for IQR - [{iqr_lower_bound}, {iqr_upper_bound}]")

        # Min - max + 99% CI
        range_mean = comparison_df.loc[1:5, "range"].mean()
        range_std = comparison_df.loc[1:5, "range"].std()
        marginal_err_range = Z_FACTOR * range_std / 4
        range_lower, range_upper = 0.44, 0.57
        #print(f"99% CI for RANGE - [{range_lower}, {range_upper}]")

        final_stat_dict_bottle["lower_bound_iqr"].append(iqr_lower_bound)
        final_stat_dict_bottle["upper_bound_iqr"].append(iqr_upper_bound)
        final_stat_dict_bottle["input_img_iqr"].append(comparison_df.loc[0, 'IQR'])
        bound_list=[iqr_lower_bound,iqr_upper_bound]
        closest = min(bound_list, key=lambda x: abs(x - comparison_df.loc[0, 'IQR']))
        print(closest,bound_list)
        if (iqr_lower_bound<=comparison_df.loc[0, 'IQR']) & (iqr_upper_bound>=comparison_df.loc[0, 'IQR']):
            lift=1
            prediction = 'Non-Defective'
        else:
            lift=abs(comparison_df.loc[0, 'IQR']-closest)+1
            prediction = 'Defective'

        final_stat_dict_bottle["lift_iqr"].append(lift)
        #final_stat_dict_bottle["lower_bound_range"].append(range_lower)
        #final_stat_dict_bottle["upper_bound_range"].append(range_upper)
        #final_stat_dict_bottle["input_img_range"].append(comparison_df.loc[0, 'range'])
        final_stat_dict_bottle["actual_lbl"].append(lbl)
        final_stat_dict_bottle["lower_bound_range"].append(0.44)
        final_stat_dict_bottle["upper_bound_range"].append(0.57)
        final_stat_dict_bottle["input_img_range"].append(comparison_df.loc[0, 'range'])

        if (range_lower <= comparison_df.loc[0, 'range']) & (range_upper >= comparison_df.loc[0, 'range']):
            final_stat_dict_bottle['prediction'].append('Non-Defective')
            final_stat_dict_bottle['prediction_result'].append(1)
            final_stat_dict_bottle['lift_range'].append(1)
        else:
            final_stat_dict_bottle['prediction'].append('Defective')
            final_stat_dict_bottle['prediction_result'].append(0)
            bound_list=[range_lower, range_upper]
            closest = min(bound_list, key=lambda x: abs(x - comparison_df.loc[0, 'range']))
            lift_range=abs((comparison_df.loc[0, 'range']-closest)/closest) * 100
            final_stat_dict_bottle['lift_range'].append(lift_range)

    final_stat_bottle_df = pd.DataFrame(final_stat_dict_bottle)
    print('final_stat_bottle_df:', final_stat_bottle_df)
    final_stat_bottle_df["result"] = (
            (final_stat_bottle_df["lower_bound_range"] <= final_stat_bottle_df["input_img_range"])
            & (final_stat_bottle_df["input_img_range"] <= final_stat_bottle_df["upper_bound_range"])
    )


    # if ((final_stat_bottle_df["lower_bound_iqr"] <= final_stat_bottle_df["input_img_iqr"])
    #     and (final_stat_bottle_df["input_img_iqr"] <= final_stat_bottle_df["upper_bound_iqr"])):
    #     final_stat_bottle_df['Prediction'] = 'Non-Defective'
    # else:
    #     final_stat_bottle_df['Prediction'] = 'Defective'
    #
    # print(final_stat_bottle_df)

    #final_stat_bottle_df["range_within_CI"] = (
    #(final_stat_bottle_df["lower_bound_range"] <= final_stat_bottle_df["input_img_range"])
    #& (final_stat_bottle_df["input_img_range"] <= final_stat_bottle_df["upper_bound_range"])
    #)

    #final_stat_bottle_df["good"] = final_stat_bottle_df["iqr_within_CI"] & final_stat_bottle_df["range_within_CI"]

    # final_stat_bottle_df.head()
    return final_stat_bottle_df, plots


def get_img_dict_mutiple(directory: Path):

    tensor_dict = {}

    for dir_ in directory.iterdir():

        dir_name = dir_.parts[-1]

        print(dir_, dir_name)

        curr_imgs = dir_



        curr_np_arr = np.array([

            np.array(Image.open(dir_).resize((256, 256)))

        ])

        curr_np_arr = curr_np_arr.astype(np.float32)

        curr_np_arr /= 255.

        img_name = curr_imgs

        print(dir_name, curr_np_arr.shape)

        tensor_dict[dir_name] = curr_np_arr
    return tensor_dict


from django.conf import settings
from keras.preprocessing import image


# def piechart(request):
#     # Pie chart, where the slices will be ordered and plotted counter-clockwise:
#     labels = 'Good', 'Bad'
#     sizes = [random.randint(10,30), random.randint(30,50)]
#     explode = (1, 1.0017)  # only "explode" the 2nd slice (i.e. 'Hogs')
#     fig1, ax1 = plt.subplots()
#     ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#             shadow=True, startangle=90)
#     ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     plt.savefig('media/good-bad-image.png', dpi=100)
#     return render(request, 'result.html')


import matplotlib.pyplot as plt
from django.conf import settings
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from plotly.offline import plot, offline
from plotly.graph_objs import Bar, pie


@login_required()
def upload_file(request):
    if request.method == 'POST':
        context = {}
        assessment_portfolio: request.session.get('assessment_portfolio')
        uploaded_file = request.FILES['file1']

        file_list = []
        for i in request.FILES.getlist('file1'):
            file_list.append(i)

        if len(file_list) is 1:
            fs = FileSystemStorage()
            name = fs.save(uploaded_file.name, uploaded_file)
            context["url"] = fs.url(name)
            testimage = '.'+context["url"]
            context['name'] = testimage
            # testimage = image.load_img(name)

            assessment_portfolio = request.session.get('assessment_portfolio')
            context["assessment_portfolio"] = assessment_portfolio
            context['test_image'] = testimage
            print(f"Imag url is {testimage}")
            bottle_imgs = get_img_dict(Path(r"./bottle/data"))
            print(bottle_imgs.keys())
            bottles_data_dict = create_in_pred_res_dict(bottle_model, bottle_imgs)
            all_merged_bottles, all_merged_bottles_labels = create_images_batch_from_dict(bottles_data_dict)
            for idx, k in enumerate(bottles_data_dict.keys()):
                f, h, r = bottles_data_dict[k]
                print(f"{idx, k}\nf - {f.shape}")
                print(f"h - {h.shape}")
                print(f"r - {r.shape}")
                print()
        else:
            fs = FileSystemStorage(location='media/test')
            for i in file_list:
                path = os.listdir(settings.MEDIA_ROOT_TEST)
                try:
                    if i.name in path:
                        # path.remove(i.name)
                       os.remove('media/test/' + i.name)
                except Exception as ex:
                    print(ex)

                fs.save(i.name, i)
                # context["url_{}".format(i)] = fs.url(fs.save(i.name, i))
                # list_of_files = fs.url(fs.save(i.name, i))
                # path = os.path.join(settings.BASE_DIR, 'media\\test\\')

                list_of_files = os.listdir('media/test')
                context['list_of_files'] = list_of_files
                print(len(list_of_files))

                # file = ('media/test/' + "url_{}".format(i))

            assessment_portfolio = request.session.get('assessment_portfolio')
            context["assessment_portfolio"] = assessment_portfolio

            # print(f"Imag url is {list_of_files}")

            bottle_imgs = get_img_dict(Path(r"./bottle/data"))
            print(bottle_imgs.keys())
            bottles_data_dict = create_in_pred_res_dict(bottle_model, bottle_imgs)
            all_merged_bottles, all_merged_bottles_labels = create_images_batch_from_dict(bottles_data_dict)
            for idx, k in enumerate(bottles_data_dict.keys()):
                f, h, r = bottles_data_dict[k]
                print(f"{idx, k}\nf - {f.shape}")
                print(f"h - {h.shape}")
                print(f"r - {r.shape}")
                print()

    # Single Images
        if assessment_portfolio == '1':
            bottle_imgs_input = get_img_dict_input(Path(rf"{testimage}"))
            bottles_data_dict_input = create_in_pred_res_dict(bottle_model, bottle_imgs_input)
            all_merged_bottles_input, all_merged_bottles_labels_input = create_images_batch_from_dict(bottles_data_dict_input)
            for idx, k in enumerate(bottles_data_dict_input.keys()):
                f_input, h_input, r_input = bottles_data_dict_input[k]
                print(f"{idx, k}\nf_input - {f_input.shape}")
                print(f"h_input - {h_input.shape}")
                print(f"r_input - {r_input.shape}")
                print()

            btl_df, btl_plots = bottles_prediction(bottle_model, all_merged_bottles[0], all_merged_bottles_input[0],
                                                   all_merged_bottles_labels_input,
                                                   all_merged_bottles_labels, bottles_data_dict, bottles_data_dict_input,
                                                   Z_FACTOR=1.96,  # 95 %CI, use 2.57 for 99% CI
                                                   )
            table = json.loads(btl_df.to_json(orient='records'))
            print(table)
            print(type(table))
            context['data'] = table[0]
            table_data = {}
            table_data['table'] = table[0]
            context['table_data'] = table_data
            print(table_data)
            context['lift_iqr'] = f"{btl_df['lift_iqr'][0]}"
        # multiple files
        else:
            bottle_imgs_dic = get_img_dict_mutiple(Path(rf"{'media/test'}"))
            bottles_data_dict_img = create_in_pred_res_dict(bottle_model, bottle_imgs_dic)
            all_merged_bottles_multiple, all_merged_bottles_labels_multiple = create_images_batch_from_dict(bottles_data_dict_img)
            for idx, k in enumerate(bottles_data_dict_img.keys()):
                f_input, h_input, r_input = bottles_data_dict_img[k]
                print(f"{idx, k}\nf_input - {f_input.shape}")
                print(f"h_input - {h_input.shape}")
                print(f"r_input - {r_input.shape}")
                print()

            btl_df, btl_plots = bottles_prediction(bottle_model, all_merged_bottles[0], all_merged_bottles_multiple[0],
                                                   all_merged_bottles_labels_multiple,
                                                   all_merged_bottles_labels, bottles_data_dict, bottles_data_dict_img,
                                                   Z_FACTOR=1.96,  # 95 %CI, use 2.57 for 99% CI
                                                   )
            table1 = json.loads(btl_df.to_json(orient='records'))
            list_of_values = []
            lower_bound = []
            upper_bound = []
            input_img = []
            prediction = []
            lift_range = []
            for i in table1:
                list_of_values.append(i.get('prediction_result'))
                request.session['list_of_values'] = list_of_values

            for i in table1:
                lower_bound.append(i.get('lower_bound_range'))
                request.session['lower_bound'] = lower_bound

            for i in table1:
                upper_bound.append(i.get('upper_bound_range'))
                request.session['upper_bound'] = upper_bound

            for i in table1:
                input_img.append(i.get('input_img_range'))
                request.session['input_img'] = input_img

            for i in table1:
                prediction.append(i.get('prediction'))
                request.session['prediction'] = prediction

            for i in table1:
                lift_range.append(i.get('lift_range'))
                request.session['lift_range'] = lift_range

            print(lower_bound)
            print(upper_bound)
            print(input_img)

            print(list_of_values)
            print(table1)
            print(type(table1))
            data = table1
            request.session['data'] = data
            table_data1 = {}
            table_data1['table1'] = table1
            context['table_data1'] = table_data1
            print(table_data1)
            # context['lift_iqr'] = f"{btl_df['lift_iqr']}"
            # context['lift_iqr'] = (btl_df.to_dict()["lift_iqr"])
            print(list_of_files)
            import matplotlib.pyplot as plt
            import numpy as np

            values =np.array(list_of_values)
            myvalues = [0, 1]
            labels = ['Non-Defective', 'Defective']

            sum_of_true = [x for x in values if x == 1]
            sum_of_false = [x for x in values if x == 0]

            addition_of_true = len(sum_of_true)
            addition_of_false = len(sum_of_false)

            fig1 = go.Figure(data=[go.Pie(labels=labels, values=[addition_of_true, addition_of_false], pull=[0.1, 0.1])])
            # fig1.update_layout(margin=dict(t=2, b=2, l=2, r=2))
            fig1.update_layout(
                autosize=False,
                width=445,
                height=450
            )
            fig1.update_traces(marker=dict(colors=['#12ABDB', '#0070AD']))
            plot_div = offline.plot(fig1, output_type='div')

            # for i in sum_of_true:
            #     addition_of_true = addition_of_true + i
            #
            # for i in sum_of_false:
            #     addition_of_false = addition_of_false + i

            total = round(addition_of_true + addition_of_false)
            y = np.array([addition_of_true, addition_of_false])
            mylabels = [str(round(addition_of_true)), str(round(addition_of_false))]
            s = ['Good', 'Bad']
            good = round(addition_of_true)
            bad = round(addition_of_false)
            plt_1 = plt.figure(figsize=(5, 5))
            colors = ['#12ABDB', '#0070AD']
            plt.pie(y, labels=mylabels, colors=colors)
            plt.legend(['Non-Defective', 'Defective'])
            plt.show()

            plt_1.savefig("media/piechart/bottle_pie")


            # fig1.show()

            # pull is given as a fraction of the pie radius
            # values =np.array(list_of_values)
            # fig1 = go.Figure(data=[go.Pie(labels=labels, values=values,, pull=[0, 0, 0.2, 0])])
            # fig1.show()
            # fig1 = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])
            # fig1 = plot([pie(labels=labels, values=values, colors=colors,)],
            #             output_type='div', image_height=50, image_width=300,)
            # trace = go.Pie(labels=labels, values=values,)
            # data = [trace]
            # fig1 = go.Figure(data = data)
            # fig1.show()

            import pandas as pd
            # x = round(list_of_values, 5)

            print('list_of_values:', list_of_values)
            print('list_of_files:', list_of_files)
            df = pd.DataFrame({
                # 'list_of_values': list_of_values,
                'Image': list_of_values,
                # 'lower_bound': lower_bound,
                'lift_range': lift_range,
                'list_of_files': list_of_files,
            })

            # fig = px.bar(df, y='lift_range', x='list_of_files', text_auto='.2s',
            #              title="Controlled text sizes, positions and angles")
            # fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)


            # fig.write_image("media/piechart/bottle_new.jpeg")
            # fig.write_image("media/bar_new", format='jpg')
            # pio.write_image(fig, "media/piechart/bottle_new.png")
            # fig.write_image("media/piechart/bottle_new", format='jpg')
            # fig.savefig("media/piechart/bottle_bar")

            col = []
            for x in list_of_values:
                if x == 1:
                    col.append('#12ABDB')
                else:
                    col.append('#0070AD')

            fig = plot([Bar(x=list_of_files, y=lift_range, marker={'color': col},
                            name='test',
                            opacity=0.8, )],
                       output_type='div', image_height=20, image_width=320,)

            df.plot(kind='bar', y='lift_range', figsize=(7, 7), color=col)
            plt.xlabel("Input Image", fontsize=14)
            plt.ylabel("Lift Range", fontsize=14)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig("media/piechart/bottle_bar")

            # import numpy as np
            # import matplotlib.pyplot as plt
            #
            # x = 'list_of_values'
            # y = 'Image'
            #
            # range1_list = [x for x in list_of_values if x == 1]
            # range2_list = [x for x in list_of_values if x == 0]
            # import ipdb
            # ipdb.set_trace()
            # col = []
            # for x in list_of_values:
            #     if x == 1:
            #         col.append('green')
            #     else:
            #         col.append('red')
            #
            #     # elif i > '1':
            #     #     col.append('red')
            #
            # # col looks like this: ['blue', 'blue', 'blue', 'blue', 'red', 'red', 'red', 'green', 'green', 'green']
            #
            # plt.bar(x, y, color=col)
            # plt.savefig("media/bar11")

            # # importing matplotlib
            # import matplotlib.pyplot
            #
            # # importing pandas as pd
            # import pandas as pd
            #
            # # importing numpy as np
            # import numpy as np
            #
            # # creating a dataframe
            # df = pd.DataFrame(np.random.rand(10, 20), columns=list_of_values)
            #
            # print(df)
            # df.plot.bar()

            input_image = request.POST.get('input_image')
            print(input_image)

            print(data)
            print('prediction:', prediction)
            print('list_of_values:', list_of_values)
            image_count = os.listdir('media/test')
            image_total = len(image_count)

            context = {
                'data': request.session.get('data'),
                'images': list_of_files,
                'lower_bound': request.session.get('lower_bound'),
                'upper_bound': request.session.get('upper_bound'),
                'input_img': request.session.get('input_img'),
                'list_of_values': request.session.get('list_of_values'),
                'total': total,
                'good': good,
                'bad': bad,
                'prediction': request.session.get('prediction'),
                'image_count':image_count,
                'fig1': fig1,
                'plot_div': plot_div,
                'fig': fig,
                # 'data1': data1,
            }

            # context['lift_iqr_1'] = f"{btl_df['lift_iqr'][1]}"
            # context['lift_iqr_2'] = f"{btl_df['lift_iqr'][2]}"
            # context['list_of_images'] = list_of_dict
        return render(request, 'result.html', context)
    else:
        return render(request, 'upload_file.html')


@login_required
def result(request):
    global images
    if request.method == 'POST':
        list_of_files = os.listdir('media/fig1/')
        data = request.session.get('data')
        print('data', data)
        print(list_of_files)
        input_image = request.POST.get('input_image')
        print(input_image)
        context = {
            'data': request.session.get('data'),
            'image': list_of_files,
            'lower_bound': request.session.get('lower_bound'),
            'upper_bound': request.session.get('upper_bound'),
            'input_img': request.session.get('input_img'),
            'list_of_values': request.session.get('list_of_values'),
            'prediction': request.session.get('prediction'),
        }
        return render(request, 'detail.html', context)
    return render(request, 'result.html')


@login_required
def test(request):
    return render(request, 'test.html')


@login_required
def detail(request):
    number = request.GET.get('search')
    context = {
        'number': number,
        'lower_bound': request.session.get('lower_bound'),
        'upper_bound': request.session.get('upper_bound'),
        'input_img': request.session.get('input_img'),
        'list_of_values': request.session.get('list_of_values'),
        'prediction': request.session.get('prediction'),
    }
    return render(request, 'detail.html', context)


@login_required
def prac(request):
    return render(request, 'prac.html')


@login_required
def logout_user(request):
    logout(request)
    return render(request, 'logout.html')