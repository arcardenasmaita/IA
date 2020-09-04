#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:39:48 2020

@author: ar
"""

import sys
import csv, operator
import importlib
import os
import pandas as pd
import datetime
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

def model_metrics(model_log_path, metric_log_path, gexp_name):
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
    #gviz.render('petrinets\\'+gexp_name+'\\petri_' + model_base + '.png')
    gviz.render('test_time\\test.png')
    #pn_visualizer.view(gviz)
    alignments_res = alignments.apply_log(metric_log, petrinet, initial_marking, final_marking, parameters=parameters)
    fitness = replay_fitness.evaluate(alignments_res, variant=replay_fitness.Variants.ALIGNMENT_BASED,
                                      parameters=parameters)
    precision = calc_precision.apply(metric_log, petrinet, initial_marking, final_marking, parameters=parameters)
    generaliz = calc_generaliz.apply(metric_log, petrinet, initial_marking, final_marking, parameters=parameters)
    #generaliz = 0
    simplic = calc_simplic.apply(petrinet)
    f_score = 2 * ((fitness['averageFitness'] * precision) / (fitness['averageFitness'] + precision))
    end_time = time()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print('Fin %02d:%02d:%02d' % (h, m, s))
    print(' F:', '%.10f' % fitness['averageFitness'], ' P:', '%.10f' % precision,
          ' FS:', '%.10f' % f_score, ' G:', '%.10f' % generaliz, ' S:', '%.10f' % simplic, ' T:',
          '%02d:%02d:%02d' % (h, m, s))
    #metrics = pd.Series([model_base, metric_base, '%.10f' % fitness['averageFitness'],
    #                     '%.10f' % precision, '%.10f' % f_score, '%.10f' % generaliz, '%.10f' % simplic,
    #                     '%02d:%02d:%02d' % (h, m, s)])
    return ""

def create_models_batch(author_name, batch_name, gexp_name):
    # author_name = 'nolle_t7912690561'
    # batch_name = 'batch1_logs_filtrados'
    # gexp_name = 'gexp13_sll_results_details'
    print('INICIO ', datetime.datetime.now().time())
    s_time = time()
    gexp = pd.read_csv(os.path.join(os.getcwd(), 'logs_filtrados', author_name, gexp_name + '.csv'))

    #csv_log_path_full = os.path.join(os.getcwd(), 'UCI_original.csv')
    for indice_fila, fila in gexp.iterrows():
        model_log_path = os.path.join(os.getcwd(), 'logs_filtrados', author_name, batch_name, fila['filename_log_filtrado'])
        #print(csv_log_path_author)
        if os.path.isfile(model_log_path):
            model = model_metrics(model_log_path, author_name[:-12]+'_' + fila['filename_log_filtrado'], gexp_name)
            print(model)
        else:
            print('**arquivo no encontrado: ', model_log_path)
    e_time = time()
    m, s = divmod(e_time - s_time, 60)
    h, m = divmod(m, 60)
    print('Tempo total: ', '%02d:%02d:%02d' % (h, m, s))
    print('FIN', datetime.datetime.now().time())


def calculate_metrics_batch(author_name, batch_name, gexp_name):
    # author_name = 'nolle_t7912690561'
    # batch_name = 'batch1_logs_filtrados'
    # gexp_name = 'gexp13_sll_results_details'
    print('INICIO ', datetime.datetime.now().time())
    s_time = time()
    #batch_number = int(batch_name[5:-15])

    gexp = pd.read_csv(os.path.join(os.getcwd(), 'logs_filtrados', author_name, gexp_name + '.csv'))
    #gexp = gexp[gexp['batch_nro'] == batch_number]

    csv_results = open('res_'+str(time())+'_'+batch_name[:-14] + '_' + gexp_name+'.csv', 'w', newline='')
    with csv_results:
        results = csv.writer(csv_results, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')
        results.writerow(
            ['grupo_experimento', 'nro_experimento', 'tipo_experimento', 'tipo_heuristica', 'funcao_f', 'funcao_g',
             'nitmax', 'nit', 'batch_size', 'early_stopping_patiente', 'early_stopping_metric', 'nro_camadas_ocultas',
             'gaussian_noise_std,dropout', 'optimizer_beta2', 'alfa', 'no', 'nome_dataset', 'model_type', 'scaling_factor',
             'limiar_EQM', 'execution_time', 'PP', 'PN', 'train_loss_last', 'batch_nro', 'filename_detection',
             'filename_log_filtrado', 'filename_log_rotulado', 'model_base', 'metric_base', 'F', 'P',
             'FS', 'G', 'S', 'time'])
        csv_log_path_full = os.path.join(os.getcwd(), 'UCI_original.csv')
        for indice_fila, fila in gexp.iterrows():
            csv_log_path_author = os.path.join(os.getcwd(), 'logs_filtrados', author_name, batch_name, fila['filename_log_filtrado'])
            #print(csv_log_path_author)
            #x = pd.read_csv(csv_log_path_author)
            if os.path.isfile(csv_log_path_author):
                metrics_filter_full = calculate_quality_metrics(csv_log_path_author, csv_log_path_full,
                                                                author_name[:-12] + '_' + fila['filename_log_filtrado'],
                                                                'full_UCI', gexp_name)
                results.writerow(pd.concat([fila, metrics_filter_full], ignore_index=True))
                print(list(pd.concat([fila, metrics_filter_full], ignore_index=True)), sep=",")
                metrics_filter_filter = calculate_quality_metrics(csv_log_path_author, csv_log_path_author,
                                                                  author_name[:-12] + '_' + batch_name[:-14] + '_' + fila['filename_log_filtrado'],
                                                                  'filter_' + author_name[:-12] + '_' + batch_name[:-14] + '_' + fila['filename_log_filtrado'],
                                                                  gexp_name)
                results.writerow(pd.concat([fila, metrics_filter_filter], ignore_index=True))
                #print(list(pd.concat([fila, metrics_filter_filter], ignore_index=True)), sep=",")
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
    # author_name = 'krugger_t7912683052'
    # batch_name = 'batch1_log_filtrados'
    # gexp_name = 'gexp12_all_results_details'
    # calculate_metrics_batch(author_name, batch_name, gexp_name)
    globals()[sys.argv[1]](sys.argv[2], sys.argv[3], sys.argv[4])

"""
    executar desde linha de comando como
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch1_log_filtrados gexp13_all_results_details
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch2_log_filtrados gexp13_all_results_details_batch2
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch1_log_filtrados nolle_gexp_batch1_part21
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch1_log_filtrados nolle_gexp_batch1_part22
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch2_log_filtrados nolle_gexp_batch2_part21
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch2_log_filtrados nolle_gexp_batch2_part22
    
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch0_log_filtrados gexp13_all_results_details_escolhidos
    
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch3_logs_filtrados krugger_gexp12_batch3_part51
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch3_logs_filtrados krugger_gexp12_batch3_part52
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch3_logs_filtrados krugger_gexp12_batch3_part53
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch3_logs_filtrados krugger_gexp12_batch3_part54
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch3_logs_filtrados krugger_gexp12_batch3_part55
    
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch0_logs_filtrados gexp12_all_results_details_escolhidos
    
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch3_logs_filtrados nolle_gexp13_batch3_part51
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch3_logs_filtrados nolle_gexp13_batch3_part52
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch3_logs_filtrados nolle_gexp13_batch3_part53
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch3_logs_filtrados nolle_gexp13_batch3_part54
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch3_logs_filtrados nolle_gexp13_batch3_part55
    
    python qualityMetricsBatch.py calculate_metrics_batch menos_frequentes batch1_logs_filtrados gexpFrequentes
    python qualityMetricsBatch.py calculate_metrics_batch menos_frequentes batch2_logs_filtrados gexp21_all_results_details_batch2
    python qualityMetricsBatch.py calculate_metrics_batch frequentes batch2_logs_filtrados frequentes_3er
    
    
    python qualityMetricsBatch.py calculate_metrics_batch aleatorio batch1_logs_filtrados gexp21_all_results_details_batch2 gexpAleatorio
    python qualityMetricsBatch.py calculate_metrics_batch aleatorio batch2_logs_filtrados gexp31_all_results_details_batch2
    
    
    qualityMetricsBatch.py : nome do script
    calculate_metrics_batch : nome do método
    nolle_t7912690561 : nome da pasta do autor
    batch3_logs_filtrados : pasta do batch de logs
    nolle_gexp13_batch3_part55: planilha de todos os logs a ser avaliados

    python qualityMetricsBatch.py calculate_metrics_batch full batch1_logs_filtrados gexp1_full_UCI
    
    #refazer
    1 python qualityMetricsBatch.py calculate_metrics_batch aalst_t7929411005 batch0_logs_filtrados gexp14_all_results_details_batch4_part1
    2 python qualityMetricsBatch.py calculate_metrics_batch aalst_t7929411005 batch0_logs_filtrados gexp14_all_results_details_batch4_part2
    3 python qualityMetricsBatch.py calculate_metrics_batch aalst_t7929411005 batch0_logs_filtrados gexp14_all_results_details_batch4_part3
    
    4 python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch0_logs_filtrados gexp13_all_results_details_batch_part1
    5 python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch0_logs_filtrados gexp13_all_results_details_batch_part2
    4 python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch0_logs_filtrados gexp13_all_results_details_batch_part3
    5 python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch0_logs_filtrados gexp13_all_results_details_batch_part4
    6 python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch0_logs_filtrados gexp13_all_results_details_batch_part5
    
    6 python qualityMetricsBatch.py calculate_metrics_batch aleatorio batch1_logs_filtrados gexp30_all_results_details_batch4
    
    
    ##########
    model
    python qualityMetricsBatch.py model_metrics model_log_path model_base gexp_name
    python qualityMetricsBatch.py create_models_batch krugger_t7912683052 batch0_logs_filtrados gexp12_krugger_for_model1007
    python qualityMetricsBatch.py create_models_batch nolle_t7912690561 batch0_logs_filtrados gexp13_nolle_for_model1007
    python qualityMetricsBatch.py create_models_batch aalst_t7929411005 batch0_logs_filtrados gexp14_aalst_for_model1007
    
    model_metrics(model_log_path, model_base, gexp_name)
    python qualityMetricsBatch.py model_metrics C:\\Users\\vcardenas.local\\qualityMetricsBatch\\logs_filtrados\\aalst_t7929411005\\batch0_logs_filtrados\\exp407_log_filtrado.csv aalst_3er_exp407_log_filtrado aalst_3er
    python qualityMetricsBatch.py model_metrics C:\\Users\\vcardenas.local\\qualityMetricsBatch\\logs_filtrados\\krugger_t7912683052\\batch0_logs_filtrados\\exp9_iter0_sf0.6_log_filtrado.csv  krugger_3er_exp9_iter0_sf0.6_log_filtrado krugger_3er 
    python qualityMetricsBatch.py model_metrics C:\\Users\\vcardenas.local\\qualityMetricsBatch\\logs_filtrados\\krugger_t7912683052\\batch0_logs_filtrados\\exp27_iter0_sf0.5_log_filtrado.csv  krugger_3er_exp27_iter0_sf0.5_log_filtrado krugger_3er 
    python qualityMetricsBatch.py model_metrics C:\\Users\\vcardenas.local\\qualityMetricsBatch\\logs_filtrados\\krugger_t7912683052\\batch0_logs_filtrados\\exp27_iter0_sf0.6_log_filtrado.csv  krugger_3er_exp27_iter0_sf0.6_log_filtrado krugger_3er 
    python qualityMetricsBatch.py model_metrics C:\\Users\\vcardenas.local\\qualityMetricsBatch\\logs_filtrados\\nolle_t7912690561\\batch0_logs_filtrados\\exp25_iter0_sf0.8_log_filtrado.csv  nolle_3er_exp25_iter0_sf0.8_log_filtrado nolle_3er
    # test
    python qualityMetricsBatch2.py model_metrics C:\\Users\\vcardenas.local\\qualityMetricsBatch\\test_time\\UCI_original.csv C:\\Users\\vcardenas.local\\qualityMetricsBatch\\test_time\\UCI_original.csv test_time
    python qualityMetricsBatch2.py model_metrics C:\\Users\\vcardenas.local\\qualityMetricsBatch\\test_time\\exp25_iter0_sf0.3_log_filtrado.csv C:\\Users\\vcardenas.local\\qualityMetricsBatch\\test_time\\exp25_iter0_sf0.3_log_filtrado.csv test_time  
    #########

Krugger
exp9_iter0_sf0.6_log_filtrado.csv
exp27_iter0_sf0.5_log_filtrado.csv
exp27_iter0_sf0.6_log_filtrado.csv

Nolle
exp25_iter0_sf0.8_log_filtrado.csv
"""