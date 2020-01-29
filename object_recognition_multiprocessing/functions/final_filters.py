from objects.homography import Homography
from objects.overlap import Overlap
from functions.rgb_histogram_matching import *
from functions.plot_manager import save_homographies
import numpy as np

import matplotlib.pyplot as plt


def create_tanh(array, index_treshold, smoothy=False):
    semi_length = len(array) / 2.0

    def custom_tanh(x):
        smoothness = 12

        if smoothy:
            smoothness = 4

        return np.round(np.tanh(smoothness / semi_length * (x - index_treshold)), 2)

    return custom_tanh


# t = np.arange(0, 20, 0.1)
# tanh=create_tanh(range(20), 5)
# plt.plot(t, tanh(t))
# plt.show()

def assign_rewards_with_claster(histogram_errors, rewards):
    rew_0 = 1
    rew_1 = .5
    rew_2 = 0
    rew_3 = -0.5
    rew_4 = -1

    for i, val in enumerate(histogram_errors):
        if 0 <= val <= 0.5:
            rewards[i] += rew_0
        elif 0.5 < val <= 1:
            rewards[i] += rew_1
        elif 1 < val <= 1.5:
            rewards[i] += rew_2
        elif 1.5 < val <= 2:
            rewards[i] += rew_3
        elif val > 2:
            rewards[i] += rew_4


def assign_rewards_with_tanh(generic_list, rewards, threshold, histogram, smoothy=False):
    indexes_list = np.arange(len(generic_list))

    generic_list, indexes_list = zip(*sorted(zip(generic_list, indexes_list), reverse=True))

    first_index_less_thr = len(generic_list) - 1

    for i, ratio in enumerate(generic_list):
        if ratio < threshold:
            first_index_less_thr = i
            break

    custom_tanh = create_tanh(generic_list, first_index_less_thr, smoothy=smoothy)

    rewards_to_add = custom_tanh(np.arange(len(generic_list)))

    j = 0
    for i in indexes_list:
        if histogram:
            # The weigths between (-1, 1) are mapped in (-5,5)
            scale = 2
            increment = np.round(scale * rewards_to_add[j], 2)
            increment = min(increment, 0)

        else:
            scale = 1
            increment = np.round(scale * rewards_to_add[j], 2)
            increment = (increment - 1) / 2  # The weigths between -1 and 1 are mapped in (-1,0)

        rewards[i] += increment
        j += 1


def overlaps_between_different_templates(homographies, test_image):
    def who_is_the_best(h1, h2):
        winner = best_homography(h1, h2, test_image)
        if winner == h2:
            return h2, h1
        elif winner == h1:
            return h1, h2
        else:
            print("Error in who is the best")
            raise Exception

    overlaps = []
    for i, h1 in enumerate(homographies):
        for j, h2 in enumerate(homographies[i + 1:]):
            if h1.polygon.overlaps(h2.polygon):
                overlaps.append(Overlap(h1, h2))

    for i in range(len(overlaps)):
        try:
            overlap = overlaps[i]

            if overlap.is_duplicate():

                best, to_remove = who_is_the_best(overlap.h1, overlap.h2)
                homographies.remove(to_remove)
                for j, over in enumerate(overlaps[i + 1:]):
                    if over.contain_to_remove(to_remove):
                        overlaps.remove(over)
        except Exception:
            pass

    return homographies


def number_of_inliers_filter(total_homographies_found: [Homography], rewards, bypass=False):
    reward_bad_ratio = -1

    if bypass:
        return

    dict_ratio = {}
    for i, h in enumerate(total_homographies_found):
        # Extract the list of ratios from the dictionary
        if h.template.name in dict_ratio.keys():
            list_ratio = dict_ratio[h.template.name]
        else:
            list_ratio = []
        # Append the new ratio
        list_ratio.append((i, h.num_inliers / h.polygon.area))  # indice, ratio

        # Re-insert the new list
        dict_ratio[h.template.name] = list_ratio

    for key in dict_ratio.keys():
        list_ratio = dict_ratio.get(key)
        list_ratio = sorted(list_ratio, key=lambda ratio: ratio[1], reverse=True)
        new_list = [(list_ratio[0][0], 0)]
        for i in range(len(list_ratio) - 1):
            new_list.append((list_ratio[i + 1][0], list_ratio[i + 1][1] / list_ratio[i][1] - 1))
        dict_ratio[key] = new_list

    for key in dict_ratio.keys():
        list_ratio = dict_ratio.get(key)
        if len(list_ratio) == 1:
            pass

        threshold_crossed = False
        for ratio in list_ratio:
            index = ratio[0]
            value = ratio[1]
            if threshold_crossed:
                rewards[index] += reward_bad_ratio
                pass
            if value < -0.6:
                rewards[index] += reward_bad_ratio
                threshold_crossed = True


def histograms_match_filter(total_homographies_found, rewards, test_image, bypass=False):
    if bypass:
        return

    histogram_errors = []
    for i, hom in enumerate(total_homographies_found):
        error_mean, _, _, _ = evaluete_homography(hom, test_image, plot=False)
        histogram_errors.append(error_mean)

    him_err = histogram_errors
    histogram_errors = np.array(histogram_errors)
    #histogram_errors = histogram_errors / histogram_errors.mean()

    # for e in sorted(histogram_errors.copy()):
    #     i = np.where(histogram_errors == e)
    #     i = i[0][0]
    #     i = histogram_errors.index(e)
    #     name = total_homographies_found[i].template.name
    #     iid = total_homographies_found[i].id_hom_global
    #     print(name, iid, e)

    # assign_rewards_with_claster(histogram_errors, rewards)
    assign_rewards_with_tanh(histogram_errors, rewards, threshold=65, histogram=True, smoothy=True)

    return him_err, np.array(him_err).mean()


def ratio_multitemplate_filter(total_homographies_found, rewards, bypass=False):
    if bypass:
        return

    ratio_list = []
    for i, h in enumerate(total_homographies_found):
        scene_area = h.polygon.area
        l1, l2 = h.template.size
        real_area = l1 * l2
        ratio = (real_area / scene_area) ** 0.5
        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)

    mean_ratio = ratio_list.mean()
    std_ratio = ratio_list.std()

    deviations_list = np.abs((ratio_list - mean_ratio) / std_ratio)

    deviations_list = np.abs(ratio_list / mean_ratio - 1)

    assign_rewards_with_tanh(deviations_list, rewards, threshold=0.5, histogram=False, smoothy=True)


def final_filter(total_homographies_found, test_image):
    if len(total_homographies_found) == 0:
        return total_homographies_found

    total_homographies_found = overlaps_between_different_templates(total_homographies_found, test_image)
    save_homographies(test_image, total_homographies_found, before_filters=True)

    names = []
    for h in total_homographies_found:
        names.append("{} [{}]".format(h.template.name, h.id_hom_global))
    names = np.array(names)

    rewards = [0] * len(total_homographies_found)
    rewards_initial = np.array(rewards)

    # Rewards -1 for bad inliers-over-area ratio
    number_of_inliers_filter(total_homographies_found, rewards)
    rewards_after_inliers = np.array(rewards)

    # Rewards between -2 ad 0
    his_err, his_err_mean = histograms_match_filter(total_homographies_found, rewards, test_image)
    rewards_after_hist = np.array(rewards)

    # Rewards between 0 and -1
    ratio_multitemplate_filter(total_homographies_found, rewards)
    rewards_after_ratio = np.array(rewards)

    for i in range(len(names)):
        print("{}: {}, {}, | {} {} |  {} ==> {}".format(names[i],
                                                        rewards_initial[i],
                                                        rewards_after_inliers[i],
                                                        round(his_err[i], 2),
                                                        round(his_err[i] / his_err_mean, 2),
                                                        rewards_after_hist[i],
                                                        rewards_after_ratio[i]))

    # Controlli
    # - overlap (1 a chi non overlappa, +0.5 a chi vince)
    # - analisi single template.. Vedere come si disribuisce il rapporto num_inliers/area -> assegnare punti
    # - istogramma
    # - ratio area (pesi con lo score ottenuto fino ad ora)

    total_homographies_found = np.array(total_homographies_found)
    total_homographies_found = total_homographies_found[np.array(rewards) > -1]
    return total_homographies_found
