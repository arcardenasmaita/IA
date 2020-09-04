#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:39:48 2020

@author: ar
"""

import datetime
import sys
import csv, operator
import importlib
import os
import pandas as pd
from time import time
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness
from pm4py.evaluation.generalization import evaluator as calc_generaliz
from pm4py.evaluation.precision import evaluator as calc_precision
from pm4py.evaluation.simplicity import evaluator as calc_simplic
from pm4py.algo.conformance.alignments import algorithm as alignments
from pm4py.objects.conversion.log import converter as log_converter


# global
# batch_name = ''

def calculate_quality_metrics(model_log_path, metric_log_path, model_base, metric_base, gexp_name):
    start_time = time()
    model_log_csv = pd.read_csv(model_log_path, ',')
    metric_log_csv = pd.read_csv(metric_log_path, ',')
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'number'}
    model_log = log_converter.apply(model_log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    metric_log = log_converter.apply(metric_log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    parameters = {inductive_miner.Variants.DFG_BASED.value.Parameters.CASE_ID_KEY: 'number',
                  inductive_miner.Variants.DFG_BASED.value.Parameters.ACTIVITY_KEY: 'incident_state',
                  inductive_miner.Variants.DFG_BASED.value.Parameters.TIMESTAMP_KEY: 'sys_updated_at',
                  alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.ACTIVITY_KEY: 'incident_state'}
    petrinet, initial_marking, final_marking = inductive_miner.apply(model_log, parameters=parameters)
    gviz = pn_visualizer.apply(petrinet, initial_marking, final_marking)
    gviz.render('petrinets\\'+gexp_name+'\\petri_csv_' + model_base + '_' + metric_base + '.png')
    #pn_visualizer.view(gviz)
    alignments_res = alignments.apply_log(metric_log, petrinet, initial_marking, final_marking, parameters=parameters)
    fitness = replay_fitness.evaluate(alignments_res, variant=replay_fitness.Variants.ALIGNMENT_BASED,
                                      parameters=parameters)
    precision = calc_precision.apply(metric_log, petrinet, initial_marking, final_marking, parameters=parameters)
    #generaliz = calc_generaliz.apply(metric_log, petrinet, initial_marking, final_marking, parameters=parameters)
    generaliz = 0
    simplic = calc_simplic.apply(petrinet)
    f_score = 2 * ((fitness['averageFitness'] * precision) / (fitness['averageFitness'] + precision))
    end_time = time()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print(model_base + '/' + metric_base + ' F:', '%.10f' % fitness['averageFitness'], ' P:', '%.10f' % precision,
          ' FS:', '%.10f' % f_score, ' G:', '%.10f' % generaliz, ' S:', '%.10f' % simplic, ' T:',
          '%02d:%02d:%02d' % (h, m, s))
    metrics = pd.Series([model_base, metric_base, '%.10f' % fitness['averageFitness'],
                         '%.10f' % precision, '%.10f' % f_score, '%.10f' % generaliz, '%.10f' % simplic,
                         '%02d:%02d:%02d' % (h, m, s)])
    return metrics


def calculate_metrics_batch(author_name, batch_name, gexp_name):
    # author_name = 'aalst_t7929411005'
    # batch_name = 'batch1_log_filtrados'
    # gexp_name = 'gexp14_all_results_details'
    print('INICIO ', datetime.datetime.now().time())
    s_time = time()
    #batch_number = int(batch_name[5:-15])
    gexp = pd.read_csv(os.path.join(os.getcwd(), 'logs_filtrados', author_name, gexp_name + '.csv'))
    #gexp = gexp[gexp['batch_nro'] == batch_number]
    csv_results = open('res_'+str(time())+'_'+author_name+'_'+batch_name[:-14] + '_' + gexp_name+'.csv', 'w', newline='')
    results = csv.writer(csv_results, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')
    results.writerow(
        ['grupo_experimento','nro_experimento','tipo_experimento','threshold','subsequence_length','nome_dataset',
         'model_type','batch_nro','TP','FN','FP','TN','PP','PN','recall','precision','fscore','execution_time_secs',
         'model_base', 'metric_base', 'F', 'P', 'FS', 'G', 'S', 'time'])
    csv_log_path_full = os.path.join(os.getcwd(), 'UCI_original.csv')
    for indice_fila, fila in gexp.iterrows():
        filename_log_filtrado = 'exp'+str(fila['nro_experimento'])+'_log_filtrado.csv'
        csv_log_path_author = os.path.join(os.getcwd(), 'logs_filtrados', author_name, batch_name, filename_log_filtrado)
        #print(csv_log_path_author)
        #x = pd.read_csv(csv_log_path_author)
        if os.path.isfile(csv_log_path_author):
            metrics_filter_full = calculate_quality_metrics(csv_log_path_author, csv_log_path_full,
                                                            'filter_model_' + author_name[:-12] + '_' + batch_name[
                                                                                                        :-14] + '_' +
                                                            filename_log_filtrado,
                                                            'full_log_UCI', gexp_name)
            results.writerow(pd.concat([fila, metrics_filter_full], ignore_index=True))
            print(list(pd.concat([fila, metrics_filter_full], ignore_index=True)), sep=",")
            metrics_filter_filter = calculate_quality_metrics(csv_log_path_author, csv_log_path_author,
                                                              'filter_model_' + author_name[:-12] + '_' + batch_name[
                                                                                                          :-14] + '_' +
                                                              filename_log_filtrado,
                                                              'filter_log_' + author_name[:-12] + '_' + batch_name[
                                                                                                        :-14] + '_' +
                                                              filename_log_filtrado, gexp_name)
            results.writerow(pd.concat([fila, metrics_filter_filter], ignore_index=True))
            print(list(pd.concat([fila, metrics_filter_filter], ignore_index=True)), sep=",")
        else:
            print('**arquivo no encontrado: ', csv_log_path_author)
    e_time = time()
    m, s = divmod(e_time - s_time, 60)
    h, m = divmod(m, 60)
    print('Tempo total: ', '%02d:%02d:%02d' % (h, m, s))
    print('FIN', datetime.datetime.now().time())
    del results
    csv_results.close()


if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2], sys.argv[3], sys.argv[4])
    # python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch2_log_filtrados gexp14_all_results_details

"""
run:

python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch1_logs_filtrados aalst_gexp_falta_batch12
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch1_logs_filtrados aalst_gexp_falta_batch13
fin:
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch1_logs_filtrados aalst_gexp_falta_batch14
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch1_logs_filtrados aalst_gexp_falta_batch11

python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch2_logs_filtrados gexp14_all_results_details_batch2

completo
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch1_logs_filtrados gexp14_all_results_details
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch2_logs_filtrados gexp14_all_results_details_batch2 

batch3
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch3_logs_filtrados aalst_gexp14_batch3_part51
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch3_logs_filtrados aalst_gexp14_batch3_part52
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch3_logs_filtrados aalst_gexp14_batch3_part53
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch3_logs_filtrados aalst_gexp14_batch3_part54
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch3_logs_filtrados aalst_gexp14_batch3_part55

escolhido
python qualityMetricsBatch_aalst.py calculate_metrics_batch aalst_t7929411005 batch0_logs_filtrados gexp14_all_results_details_escolhidos
"""
