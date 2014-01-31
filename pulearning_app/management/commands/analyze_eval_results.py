# -*- coding: utf-8 -*-

"""
This custom management command loads all the classification results
obtained so far and asks the user which parameters they want to compare.
The respective results with the chosen parameter values are then compared and
sorted by the different evaluation criteria. If Matplotlib is installed, the
precision-recall curves and Receiver Operating Characteristic (ROC) curves
are plotted to one figure which simplifies the comparison of the results.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


from collections import OrderedDict
import csv
import os
import re

from django.conf import settings
from django.core.management.base import BaseCommand
import matplotlib.pyplot as plt
import numpy as np


class Command(BaseCommand):

    """
    This command compares and analyzes evaluation results that are chosen
    on the command line.

    """


    def _pred_func(self, predicates, eval_params=None):
        """Filter the results by certain parameters that should be compared."""
        if eval_params is None:
            eval_params = {}

        def func((i, d)):
            if eval_params and isinstance(eval_params, dict):
                return all(eval_params[i][param] == val for param, val in predicates.iteritems())
            else:
                return all(d[param] == val for param, val in predicates.iteritems())

        return func


    def _sort_func(self, sort_keys):
        """Sort the results by the selected evaluation method."""
        def func((i, d)):
            return [-d[sort_key[1:]] if sort_key.startswith('-')
                    else d[sort_key] for sort_key in sort_keys]

        return func


    def _load_eval_results(self):
        """Load all evaluation results obtained so far into memory."""

        # Path to directory evaluation_results
        eval_results_path = os.path.join(settings.ROOT_PATH, 'evaluation_results')

        # Dictionaries holding evaluation parameters and results
        filenames = {}
        eval_params = {}
        eval_results = {}
        prec_rec_values = {}
        roc_values = {}

        # Pattern for splitting param names and values in txt file
        split_pattern = re.compile(r'\s{2,}')

        for root, dirs, files in os.walk(eval_results_path):

            # Ignore root folder and subdirectories with 'PROBLEM' in dir name
            if not root.endswith('evaluation_results') and not 'PROBLEM' in root and not root.endswith(('1dnf', 'cr', 'nb', 'rocchio', 'spy')):

                # Get the unique results_id for a specific parameter combination
                results_id = int(root.split('/')[-1].split('_', 1)[0])

                for d in (filenames, eval_params, eval_results, prec_rec_values, roc_values):
                    d[results_id] = {}

                for filename in files:

                    # Save plot values in Python lists
                    if filename.endswith('.csv'):
                        csv_path = os.path.join(eval_results_path, root, filename)
                        tmp_lst = []
                        for col1, col2 in csv.reader(open(csv_path, 'rb'), delimiter='\t'):
                            if not col1.startswith('#'):
                                tmp_lst.append((col1, col2))

                        if filename == 'precision_recall_values.csv':
                            prec_rec_values[results_id] = tmp_lst
                        elif filename == 'roc_values.csv':
                            roc_values[results_id] = tmp_lst

                    # Find the txt file that contains the evaluation results
                    elif filename.endswith('.txt'):

                        filenames[results_id] = filename[:-4]

                        # Remove suffix .txt and
                        # split filename into parameter substrings
                        params = filename[:-4].split('_')
                        for param in params:

                            # Split parameters in param_name and param_value
                            param_name, param_value = param.split('-', 1)

                            # Convert numerical parameter values
                            # from string to float
                            try:
                                param_value = float(param_value)
                            except ValueError:
                                pass

                            # Add parameters to dictionary
                            eval_params[results_id][param_name] = param_value

                        # Create full path to txt file
                        txt_path = os.path.join(eval_results_path, root, filename)

                        # Open the txt file, iterate through the lines and
                        # save evaluation results in dictionary
                        with open(txt_path) as fobj:
                            for line in fobj:
                                line = line.strip()

                                if line.startswith('Documents as Input for Reliable Negatives Detection'):
                                    eval_results[results_id]['rn_input_docs'] = int(split_pattern.split(line)[-1])

                                elif line.startswith('Documents Annotated as Reliable Negative'):
                                    eval_results[results_id]['rn_total'] = int(split_pattern.split(line)[-1].split()[0])

                                elif line.startswith('Documents as Input for Final Labeling:'):
                                    eval_results[results_id]['clf_input_docs'] = int(split_pattern.split(line)[-1])

                                elif line.startswith('Documents Annotated as Positive'):
                                    eval_results[results_id]['pos_total'] = int(split_pattern.split(line)[-1].split()[0])

                                elif line.startswith('Documents Annotated as Negative'):
                                    eval_results[results_id]['neg_total'] = int(split_pattern.split(line)[-1].split()[0])

                                elif line.startswith('Accuracy Score'):
                                    eval_results[results_id]['acc_score'] = float(split_pattern.split(line)[-1])

                                elif line.startswith('Precision Score'):
                                    eval_results[results_id]['pos_prec_score'] = float(split_pattern.split(line)[-1])

                                elif line.startswith('Average Precision Score'):
                                    eval_results[results_id]['avg_prec_score'] = float(split_pattern.split(line)[-1])

                                elif line.startswith('Recall Score'):
                                    eval_results[results_id]['pos_rec_score'] = float(split_pattern.split(line)[-1])

                                elif line.startswith('Matthews Correlation (=Phi) Coefficient'):
                                    eval_results[results_id]['matt_corrcoef'] = float(split_pattern.split(line)[-1])

                                elif line.startswith('Area under Precision-Recall Curve'):
                                    eval_results[results_id]['pr_auc_score'] = float(split_pattern.split(line)[-1])

                                elif line.startswith('Area under ROC Curve'):
                                    eval_results[results_id]['roc_auc_score'] = float(split_pattern.split(line)[-1])

                                elif line.startswith('-1'):
                                    neg_scores = line.split()
                                    eval_results[results_id]['neg_prec_score'] = float(neg_scores[1])
                                    eval_results[results_id]['neg_rec_score'] = float(neg_scores[2])
                                    eval_results[results_id]['neg_f1_score'] = float(neg_scores[3])

                                elif line.startswith('1'):
                                    pos_scores = line.split()
                                    eval_results[results_id]['pos_f1_score'] = float(pos_scores[3])

                                elif line.startswith('avg / total'):
                                    avg_scores = split_pattern.split(line)
                                    eval_results[results_id]['total_prec_score'] = float(avg_scores[1])
                                    eval_results[results_id]['total_rec_score'] = float(avg_scores[2])
                                    eval_results[results_id]['total_f1_score'] = float(avg_scores[3])

        # Convert NGRAM values from '1-1' to (1,1) etc.
        # Convert STEM and SVMIT values from 'False' to False etc.
        for results_id in eval_params:
            #ngram_value = eval_params[results_id]['NGRAM']
            #stem_value = eval_params[results_id]['STEM']
            #eval_params[results_id]['NGRAM'] = tuple((int(i) for i in ngram_value.split('-')))
            #eval_params[results_id]['STEM'] = stem_value[0] == 'T'

            # The parameters 'EM' and 'SVMIT' do not appear in all settings
            if not 'EM' in eval_params[results_id]:
                eval_params[results_id]['EM'] = 'False'
            if not 'SVMIT' in eval_params[results_id]:
                eval_params[results_id]['SVMIT'] = 'False'

        # Convert plot value lists to numpy arrays of correct shape
        # First row to be plotted on y-axis, second row on x-axis
        for d in (prec_rec_values, roc_values):
            for results_id in d:
                plot_values = d[results_id]
                plot_values = np.array(plot_values).astype(np.float).T
                d[results_id] = plot_values

        return eval_params, eval_results, prec_rec_values, roc_values, filenames


    def handle(self, *args, **options):
        """"""
        # Load the results first
        eval_params, eval_results, prec_rec_values, roc_values, filenames = self._load_eval_results()

        # Create set of all valid parameters that
        # will be used for checking user input
        valid_params = set()
        for result_id in eval_params:
            for param in eval_params[result_id]:
                valid_params.add(param)
        valid_params_dict = {}
        for i, param in enumerate(sorted(valid_params)):
            valid_params_dict[i+1] = param

        # Dictionary for holding entered parameters
        chosen_params = {}

        # Boolean for checking whether user has quit the program
        has_quit = False

        # Create separate lists of tuples sorted by certain criteria
        results_sorted_by_acc_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['acc_score'], reverse=True)

        results_sorted_by_pos_prec_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['pos_prec_score'], reverse=True)

        results_sorted_by_avg_prec_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['avg_prec_score'], reverse=True)

        results_sorted_by_pos_rec_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['pos_rec_score'], reverse=True)

        results_sorted_by_matthews_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['matt_corrcoef'], reverse=True)

        results_sorted_by_pr_auc_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['pr_auc_score'], reverse=True)

        results_sorted_by_roc_auc_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['roc_auc_score'], reverse=True)

        results_sorted_by_neg_prec_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['neg_prec_score'], reverse=True)

        results_sorted_by_neg_rec_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['neg_rec_score'], reverse=True)

        results_sorted_by_neg_f1_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['neg_f1_score'], reverse=True)

        results_sorted_by_pos_f1_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['pos_f1_score'], reverse=True)

        results_sorted_by_total_prec_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['total_prec_score'], reverse=True)

        results_sorted_by_total_rec_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['total_rec_score'], reverse=True)

        results_sorted_by_total_f1_score = sorted(
            eval_results.iteritems(), key=lambda (i, d): d['total_f1_score'], reverse=True)

        result_lists = [
            results_sorted_by_acc_score,
            results_sorted_by_pos_prec_score,
            results_sorted_by_avg_prec_score,
            results_sorted_by_pos_rec_score,
            results_sorted_by_matthews_score,
            results_sorted_by_pr_auc_score,
            results_sorted_by_roc_auc_score,
            results_sorted_by_neg_prec_score,
            results_sorted_by_neg_rec_score,
            results_sorted_by_neg_f1_score,
            results_sorted_by_pos_f1_score,
            results_sorted_by_total_prec_score,
            results_sorted_by_total_rec_score,
            results_sorted_by_total_f1_score
        ]

        result_measures = [
            'acc_score',
            'pos_prec_score',
            'avg_prec_score',
            'pos_rec_score',
            'matt_corrcoef',
            'pr_auc_score',
            'roc_auc_score',
            'neg_prec_score',
            'neg_rec_score',
            'neg_f1_score',
            'pos_f1_score',
            'total_prec_score',
            'total_rec_score',
            'total_f1_score'
        ]

        print '\nEVALUATION RESULTS ANALYSIS'
        print '===========================\n'

        print 'Decide how you want to filter your data.'
        print 'Type "quit" at every time to quit the program.\n'

        do_filtering = ''
        while True:
            do_filtering = raw_input('Do you want to filter your data by certain parameters? [Y/N]: ')

            if do_filtering == 'quit':
                has_quit = True
                break

            if do_filtering in ('Y', 'y', 'N', 'n'):
                break
            else:
                print '\nINVALID choice, please try again\n'


        if do_filtering in ('Y', 'y'):

            while True:

                print 'You can filter by the following parameters:\n'
                for i, param in sorted(valid_params_dict.iteritems()):
                    print str(i) + ':', param

                param_number = raw_input('Enter number of parameter to filter by: ')

                if param_number == 'quit':
                    has_quit = True
                    break

                try:
                    param_number = int(param_number)
                except ValueError:
                    pass

                if not param_number in valid_params_dict:
                    print '\nINVALID parameter, please try again\n'
                    continue

                param_value = raw_input('Value for chosen parameter: ')

                if param_value == 'quit':
                    has_quit = True
                    break

                # Put chosen param value combination in dictionary
                chosen_params[valid_params_dict[param_number]] = param_value

                # Delete previous choice so that it cannot be selected again
                del valid_params_dict[param_number]

                print 'You have entered the following parameters so far:'
                for param_name in chosen_params:
                    print 'PARAM:', param_name, '     VALUE:', chosen_params[param_name]

                decision = ''
                while True:
                    decision = raw_input('\nWould you like to enter more parameters? [Y/N]: ')

                    if decision == 'quit':
                        has_quit = True
                        break

                    if decision in ('Y', 'y', 'N', 'n'):
                        break
                    else:
                        print '\nINVALID choice, please try again\n'

                if has_quit or decision in ('N', 'n'):
                    break


        if not has_quit:

            while True:

                top_n = raw_input('How many top n results to output at maximum?: ')

                if top_n == 'quit':
                    has_quit = True
                    break

                try:
                    top_n = int(top_n)
                except ValueError:
                    print '\nINVALID number, please try again\n'
                    continue

                break


        if not has_quit:

            # Path to directory evaluation_results_summaries
            eval_summaries_path = os.path.join(settings.ROOT_PATH, 'evaluation_summaries')

            # File name of output subdirectory
            if chosen_params:
                subdirectory = '_'.join(['-'.join((key, val)) for key, val in sorted(chosen_params.iteritems())])
            else:
                subdirectory = 'total_best_results'

            # Path to output directory
            subdirectory_path = os.path.join(eval_summaries_path, subdirectory)

            # Create subdirectory if it does not exist
            if not os.path.exists(subdirectory_path):
                os.makedirs(subdirectory_path)

            pr_auc_values = OrderedDict()
            roc_auc_values = OrderedDict()

            # Open file and write output to it
            with open(os.path.join(subdirectory_path, subdirectory + '_top-' + str(top_n) + '.txt'), 'w') as fobj:

                for i, result_list in enumerate(result_lists):
                    result_measure = result_measures[i]
                    print >> fobj, 'MEASURE:', result_measure

                    if do_filtering in ('Y', 'y'):
                        results = filter(self._pred_func(chosen_params, eval_params), OrderedDict(result_list).iteritems())[:top_n]
                    else:
                        results = result_list[:top_n]

                    for entry in results:
                        entry_id = entry[0]
                        entry_val = entry[1][result_measure]

                        print >> fobj, 'ID:', entry_id, '  -  ', 'VAL:', entry_val, '  -  ', 'FILE PARAMS:', filenames[entry_id]

                        if result_measure == 'pr_auc_score':
                            pr_auc_values[entry_id] = entry_val
                        elif result_measure == 'roc_auc_score':
                            roc_auc_values[entry_id] = entry_val

                    print >> fobj, '\n'

                print 'File', subdirectory + '_top-' + str(top_n) + '.txt', 'written successfully!'


            # ---------------------- Plotting Statistics --------------------------

            line_styles = (
                # Real line styles
                '-', '-.', ':',
                # Markers for each data point
                '1', '2', '3', '4', 'o', 'v', '^', '<', '>', 's', 'p', '*', '+', 'x', 'd', '|', '_'
            )

            print 'Plotting precision against recall...'
            plt.figure(0, figsize=(10, 6))
            ax = plt.subplot(111)
            if chosen_params:
                plt.title(
                    'Parameters: ' + ', '.join(['(' + '-'.join((key, val)) + ')' for key, val in sorted(chosen_params.iteritems())]),
                    family='sans-serif',
                    fontweight='bold',
                    fontsize=7
                )
            else:
                plt.title(
                    'Total best results',
                    family='sans-serif',
                    fontweight='bold'
                )
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.grid(True)

            for i, result_id in enumerate(pr_auc_values):

                precision_vals, recall_vals = prec_rec_values[result_id]

                ax.plot(
                    recall_vals,
                    precision_vals,
                    'k'+line_styles[i],
                    label=str(pr_auc_values[result_id])  + '  |  ' + filenames[result_id],
                    linewidth=2,
                )

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])

            plt.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.12), frameon=False)
            plt.savefig(os.path.join(subdirectory_path, 'plot-pr_' + subdirectory + '_top-' + str(top_n) + '.pdf'))
            print 'Precision-Recall plot saved successfully!'


            print 'Plotting ROC curve...'
            plt.figure(1, figsize=(10, 6))
            ax = plt.subplot(111)
            if chosen_params:
                plt.title(
                    'Parameters: ' + ', '.join(['(' + '-'.join((key, val)) + ')' for key, val in sorted(chosen_params.iteritems())]),
                    family='sans-serif',
                    fontweight='bold',
                    fontsize=7
                )
            else:
                plt.title(
                    'Total best results',
                    family='sans-serif',
                    fontweight='bold'
                )
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.grid(True)
            ax.plot([0,1], [0,1], 'k--')

            for i, result_id in enumerate(roc_auc_values):

                tpr, fpr = roc_values[result_id]

                ax.plot(
                    fpr,
                    tpr,
                    'k'+line_styles[i],
                    label=str(roc_auc_values[result_id]) + '  |  ' + filenames[result_id],
                    linewidth=2,
                )

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])

            plt.legend(loc='upper center', fontsize=8, bbox_to_anchor=(0.5, -0.12), frameon=False)
            plt.savefig(os.path.join(subdirectory_path, 'plot-roc_' + subdirectory + '_top-' + str(top_n) + '.pdf'))

            print 'ROC plot saved successfully!'


        if has_quit:
            print '\nYou have CANCELLED the program. Bye bye. :-)\n'

