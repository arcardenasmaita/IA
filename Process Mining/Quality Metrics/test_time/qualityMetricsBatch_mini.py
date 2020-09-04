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

def calculate_quality_metrics(model_log_path, metric_log_path, model_base, metric_base):
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
    gviz.render('petrinets/petri_csv_' + model_base + '_' + metric_base + '.png')
    # pn_visualizer.view(gviz)
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
    #print(model_base + '/' + metric_base + ' F:', '%.10f' % fitness['averageFitness'], ' P:', '%.10f' % precision,
    #      ' FS:', '%.10f' % f_score, ' G:', '%.10f' % generaliz, ' S:', '%.10f' % simplic, ' T:',
    #      '%02d:%02d:%02d' % (h, m, s))
    metrics = pd.Series([model_base, metric_base, '%.10f' % fitness['averageFitness'],
                         '%.10f' % precision, '%.10f' % f_score, '%.10f' % generaliz, '%.10f' % simplic,
                         '%02d:%02d:%02d' % (h, m, s)])
    print(list(metrics), sep=",")
    return metrics
    # csv_log_path_full = '/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsPM4PYSara/UCI_incident_event_log_IDEAL_v01-log-com-anomalias-CompletoTimeStamp.csv'
    # csv_log_path_aalst = '/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsPM4PYSara/UCI_incident_event_log_IDEAL_v01-log-com-anomalias-FiltradoAalstTimeStamp.csv'
    # csv_log_path_nolle = '/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsPM4PYSara/UCI_incident_event_log_IDEAL_v01-log-com-anomalias-FiltradoNolleTimeStamp.csv'
    # csv_log_path_krugger = '/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsPM4PYSara/UCI_incident_event_log_IDEAL_v01-log-com-anomalias-FiltradoEstherTimeStamp.csv'
    # csv_log_path_less_freq = '/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsPM4PYSara/UCI_incident_event_log_IDEAL_v01-log-com-anomalias-FiltradoMenosFrequentes800TimeStamp.csv'
    # csv_log_path_aleatory = '/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsPM4PYSara/UCI_incident_event_log_IDEAL_v01-log-com-anomalias-FiltradoAleatorio100TimeStamp.csv'
    # calculate_quality_metrics(csv_log_path_full, csv_log_path_full, 'full', 'full')
    # calculate_quality_metrics(csv_log_path_aalst, csv_log_path_full, 'aalst', 'full')
    # calculate_quality_metrics(csv_log_path_nolle, csv_log_path_full, 'nole', 'full')
    # calculate_quality_metrics(csv_log_path_krugger, csv_log_path_full, 'krugger', 'full')
    # calculate_quality_metrics(csv_log_path_less_freq, csv_log_path_full, 'less_freq', 'full')
    # calculate_quality_metrics(csv_log_path_aleatory, csv_log_path_full, 'aleatory', 'full')
    # calculate_quality_metrics(csv_log_path_aalst, csv_log_path_aalst, 'aalst', 'aalst')
    # calculate_quality_metrics(csv_log_path_nolle, csv_log_path_nolle, 'nole', 'nole')
    # calculate_quality_metrics(csv_log_path_krugger, csv_log_path_krugger, 'krugger', 'krugger')
    # calculate_quality_metrics(csv_log_path_less_freq, csv_log_path_less_freq, 'less_freq', 'less_freq')
    # calculate_quality_metrics(csv_log_path_aleatory, csv_log_path_aleatory, 'aleatory', 'aleatory')


def calculate_metrics_batch(author_name, batch_name, gexp_name):
    print(datetime.datetime.now().time())

    batch_number = int(batch_name[5:-14])

    gexp = pd.read_csv(os.path.join(os.getcwd(), 'logs_filtrados', author_name, gexp_name + '.csv'))
    gexp = gexp[gexp['batch_nro'] == batch_number]

    results = pd.DataFrame(columns = ('grupo_experimento', 'nro_experimento', 'tipo_experimento', 'tipo_heuristica', 'funcao_f', 'funcao_g',
         'nitmax', 'nit', 'batch_size', 'early_stopping_patiente', 'early_stopping_metric', 'nro_camadas_ocultas',
         'gaussian_noise_std,dropout', 'optimizer_beta2', 'alfa', 'no', 'nome_dataset', 'model_type', 'scaling_factor',
         'limiar_EQM', 'execution_time', 'PP', 'PN', 'train_loss_last', 'batch_nro', 'filename_detection',
         'filename_log_filtrado', 'filename_log_rotulado', 'model_base', 'metric_base', 'F', 'P',
         'FS', 'G', 'S', 'time'))
    csv_log_path_full = os.path.join(os.getcwd(), 'UCI_original.csv')
    for indice_fila, fila in gexp.iterrows():
        # csv_log_path_author = '/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsBatch/logs_filtrados/nolle_t7912690561/batch1_log_filtrados/exp10_iter0_sf0.0_log_filtrado.csv'
        csv_log_path_author = os.path.join(os.getcwd(), 'logs_filtrados', author_name, batch_name,
                                           fila['filename_log_filtrado'])
        #print(csv_log_path_author)
        #x = pd.read_csv(csv_log_path_author)
        if os.path.isfile(csv_log_path_author):
            metrics_filter_full = calculate_quality_metrics(csv_log_path_author, csv_log_path_full,
                                                            'filter_model_' + author_name[:-12] + '_' + batch_name[:-14] + '_' + gexp_name[:-20],
                                                            'full_log_UCI')
            fila_completa = pd.Series(pd.concat([fila, metrics_filter_full], ignore_index=True))
            results.append(fila_completa, ignore_index=True)
            metrics_filter_filter = calculate_quality_metrics(csv_log_path_author, csv_log_path_author,
                                                              'filter_model_' + author_name[:-12] + '_' + batch_name[:-14] + '_' + gexp_name[:-20],
                                                              'filter_log_' + author_name[:-12] + '_' + batch_name[:-14] + '_' + gexp_name[:-20])
            fila_completa = pd.Series(pd.concat([fila, metrics_filter_filter], ignore_index=True))
            results.append(fila_completa, ignore_index=True)
        else:
            print('**arquivo no encontrado: ', csv_log_path_author)
        #print('time exp line: ', datetime.datetime.now().time())
    
    results.to_csv('res_'+str(time())+'_'+batch_name[:-14] + '_' + gexp_name+'.csv', index = False, header = True)        
    print('time finish!! ', datetime.datetime.now().time())
        # if os.path.isfile(csv_log_path_author):
        #     metrics_filter_full = calculate_quality_metrics(csv_log_path_author, csv_log_path_full,
        #                                                     'filter_model_' + author_name[:-12] + '_' + batch_name[:-14] + '_' + gexp_name[:-20],
        #                                                     'full_log_UCI')
        #     results.writerow(pd.concat([fila, metrics_filter_full], ignore_index=True))
        #     metrics_filter_filter = calculate_quality_metrics(csv_log_path_author, csv_log_path_author,
        #                                                       'filter_model_' + author_name[:-12] + '_' + batch_name[:-14] + '_' + gexp_name[:-20],
        #                                                       'filter_log_' + author_name[:-12] + '_' + batch_name[:-14] + '_' + gexp_name[:-20])
        #     results.writerow(pd.concat([fila, metrics_filter_filter], ignore_index=True))
        # else:
        #     print('**arquivo no encontrado: ', csv_log_path_author)

    

if __name__ == '__main__':
    # author_name = 'nolle_t7912690561'
    # batch_name = 'batch1_log_filtrados'
    # gexp_name = 'nolle_gexp_falta_batch101'
    # calculate_metrics_batch(author_name, batch_name, gexp_name)
    globals()[sys.argv[1]](sys.argv[2], sys.argv[3], sys.argv[4])
    # executar desde linha de comando como
    # python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch1_log_filtrados gexp13_all_results_details
    # python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch2_log_filtrados gexp13_all_results_details_batch2
    # python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch1_log_filtrados gexp12_all_results_details
    # python qualityMetricsBatch_mini.py calculate_metrics_batch nolle_t7912690561 batch1_log_filtrados nolle_gexp_falta_batch101
    #

    #python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch1_log_filtrados aalst_gexp_falta_batch2_all