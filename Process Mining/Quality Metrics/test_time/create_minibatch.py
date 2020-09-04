#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 23:12:36 2020

@author: ar
"""
import pandas as pd

###### ---------- AALST
aalst_avance  =  pd.read_csv('/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsBatch/results_aalst_t7929411005_batch2_gexp14_all_re_avance1652.csv')
aalst_gexp  =  pd.read_csv('/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsBatch/logs_filtrados/aalst_t7929411005/gexp14_all_results_details_batch2.csv')

aalst_list_avance = list(aalst_avance['nro_experimento'].drop_duplicates())

aalst_gexp = aalst_gexp[aalst_gexp['batch_nro'].isin([2])]
aalst_gexp_falta = aalst_gexp[~aalst_gexp['nro_experimento'].isin(aalst_list_avance)]

aalst_gexp_falta.to_csv('aalst_t7929411005_batch2_gexp14_falta', index = False, header = True)


# anterior
gexp_batch11 = aalst_gexp_falta[:10]
gexp_batch11.to_csv('aalst_gexp_falta_batch11', index = False, header = True)
gexp_batch12 = aalst_gexp_falta[10:20]
gexp_batch12.to_csv('aalst_gexp_falta_batch12', index = False, header = True)
gexp_batch13 = aalst_gexp_falta[20:30]
gexp_batch13.to_csv('aalst_gexp_falta_batch13', index = False, header = True)
gexp_batch14 = aalst_gexp_falta[30:]
gexp_batch14.to_csv('aalst_gexp_falta_batch14', index = False, header = True)

###### ---------- NOLLE

nolle_avance  =  pd.read_csv('/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsBatch/results_nolle_t7912690561_batch1_gexp13_avance1144.csv')
nolle_gexp  =  pd.read_csv('/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsBatch/logs_filtrados/nolle_t7912690561/gexp13_all_results_details.csv')

nolle_list_exp_avance = list(nolle_avance['filename_log_filtrado'].drop_duplicates())
nolle_gexp = nolle_gexp[nolle_gexp['batch_nro'].isin([1])]
nolle_gexp_falta = nolle_gexp[~nolle_gexp['filename_log_filtrado'].isin(nolle_list_exp_avance)]

gexp_batch101 = nolle_gexp_falta[:20]
gexp_batch101.to_csv('nolle_gexp_falta_batch101.csv', index = False, header = True)
gexp_batch102 = nolle_gexp_falta[20:40]
gexp_batch102.to_csv('nolle_gexp_falta_batch102.csv', index = False, header = True)
gexp_batch103 = nolle_gexp_falta[40:60]
gexp_batch103.to_csv('nolle_gexp_falta_batch103.csv', index = False, header = True)
gexp_batch104 = nolle_gexp_falta[60:80]
gexp_batch104.to_csv('nolle_gexp_falta_batch104.csv', index = False, header = True)
gexp_batch105 = nolle_gexp_falta[80:100]
gexp_batch105.to_csv('nolle_gexp_falta_batch105.csv', index = False, header = True)
gexp_batch106 = nolle_gexp_falta[100:120]
gexp_batch106.to_csv('nolle_gexp_falta_batch106.csv', index = False, header = True)
gexp_batch107 = nolle_gexp_falta[120:140]
gexp_batch107.to_csv('nolle_gexp_falta_batch107.csv', index = False, header = True)
gexp_batch108 = nolle_gexp_falta[140:160]
gexp_batch108.to_csv('nolle_gexp_falta_batch108.csv', index = False, header = True)
gexp_batch109 = nolle_gexp_falta[160:180]
gexp_batch109.to_csv('nolle_gexp_falta_batch109.csv', index = False, header = True)
gexp_batch110 = nolle_gexp_falta[180:]
gexp_batch110.to_csv('nolle_gexp_falta_batch110.csv', index = False, header = True)

nolle_avance  =  pd.read_csv('/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsBatch/results_nolle_t7912690561_batch2_gexp13_all_re_avance1043.csv')
nolle_gexp  =  pd.read_csv('/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsBatch/logs_filtrados/nolle_t7912690561/gexp13_all_results_details_batch2.csv')

nolle_list_exp_avance = list(nolle_avance['filename_log_filtrado'].drop_duplicates())
nolle_gexp = nolle_gexp[nolle_gexp['batch_nro'].isin([1])]
nolle_gexp_falta = nolle_gexp[~nolle_gexp['filename_log_filtrado'].isin(nolle_list_exp_avance)]

nolle_gexp.to_csv('nolle_gexp_falta_batch2_all', index = False, header = True)


# divide nolle completo batch 1 en 2
nolle_gexp  =  pd.read_csv('/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsBatch/logs_filtrados/nolle_t7912690561/gexp13_all_results_details_batch2.csv')
nolle_gexp = nolle_gexp[nolle_gexp['batch_nro'].isin([2])]
gexp_batch21 = nolle_gexp[:135]
gexp_batch21.to_csv('nolle_gexp_batch2_part21.csv', index = False, header = True)
gexp_batch22 = nolle_gexp[135:]
gexp_batch22.to_csv('nolle_gexp_batch2_part22.csv', index = False, header = True)


nolle_gexp  =  pd.read_csv('/media/ar/DATA/anomaly_detection_artigo/medidas_avaliacao/qualityMetricsBatch/logs_filtrados/nolle_t7912690561/gexp13_all_results_details.csv')
nolle_gexp = nolle_gexp[nolle_gexp['batch_nro'].isin([1])]
gexp_batch21 = nolle_gexp[:150]
gexp_batch21.to_csv('nolle_gexp_batch1_part11.csv', index = False, header = True)
gexp_batch22 = nolle_gexp[150:]
gexp_batch22.to_csv('nolle_gexp_batch1_part12.csv', index = False, header = True)

gexp_batch21_nolle = nolle_gexp_falta[180:]
nolle_gexp.to_csv('nolle_gexp_falta_batch2_all', index = False, header = True)

#-- nolle batch4
gexp  =  pd.read_csv(gexp13_all_results_details_batch4)
gexp_batch21 = nolle_gexp[:150]
gexp_batch21.to_csv('nolle_gexp_batch1_part11.csv', index = False, header = True)
gexp_batch22 = nolle_gexp[150:]
gexp_batch22.to_csv('nolle_gexp_batch1_part12.csv', index = False, header = True)


#------ kruger

avance  =  pd.read_csv'results_krugger_t7912683052_batch1_sara.csv')
gexp  =  pd.read_csv('results_ateriores/gexp12_all_results_details_batch1.csv')

list_exp_avance = list(avance['filename_log_filtrado'].drop_duplicates())
gexp = gexp[gexp['batch_nro'].isin([1])]
gexp_falta = gexp[~gexp['filename_log_filtrado'].isin(list_exp_avance)]

gexp_part = gexp_falta[:10]
gexp_part.to_csv('krugger_gexp12_batch1_falta_de_sara_part31_ok.csv', index = False, header = True)
gexp_part = gexp_falta[10:20]
gexp_part.to_csv('krugger_gexp12_batch1_falta_de_sara_part32_ok.csv', index = False, header = True)
gexp_part = gexp_falta[20:]
gexp_part.to_csv('krugger_gexp12_batch1_falta_de_sara_part33_ok.csv', index = False, header = True)

"""
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch1_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part31_ok
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch1_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part32_ok
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch1_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part33_ok
    
    -- batch1_parte31 -> menos um: exp25_iter0_sf0.3_log_filtrado.csv
    python qualityMetricsBatch_um.py calculate_metrics_batch krugger_t7912683052 batch1_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part31_ok_um
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch1_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part31_ok_menosum_21
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch1_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part31_ok_menosum_21_42
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch1_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part31_ok_menosum_21_43
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch1_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part31_ok_menosum_21_44
    python qualityMetricsBatch.py calculate_metrics_batch krugger_t7912683052 batch1_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part31_ok_menosum_22
    
    -- batch4
    python qualityMetricsBatch.py calculate_metrics_batch nolle_t7912690561 batch4_logs_filtrados krugger_gexp12_batch1_falta_de_sara_part31_ok_menosum_22
"""
# ----------- batch3
gexp = pd.read_csv('results_ateriores/gexp12_all_res_batch3_krugger.csv')
gexp = gexp[gexp['batch_nro'].isin([3])]
gexp_part = gexp[:50]
gexp_part.to_csv('krugger_gexp12_batch3_part51.csv', index = False, header = True)
gexp_part = gexp[50:100]
gexp_part.to_csv('krugger_gexp12_batch3_part52.csv', index = False, header = True)
gexp_part = gexp[100:150]
gexp_part.to_csv('krugger_gexp12_batch3_part53.csv', index = False, header = True)
gexp_part = gexp[150:200]
gexp_part.to_csv('krugger_gexp12_batch3_part54.csv', index = False, header = True)
gexp_part = gexp[200:]
gexp_part.to_csv('krugger_gexp12_batch3_part55.csv', index = False, header = True)

gexp = pd.read_csv('results_ateriores/gexp13_all_res_batch3_nolle.csv')
gexp = gexp[gexp['batch_nro'].isin([3])]
gexp_part = gexp[:50]
gexp_part.to_csv('nolle_gexp13_batch3_part51.csv', index = False, header = True)
gexp_part = gexp[50:100]
gexp_part.to_csv('nolle_gexp13_batch3_part52.csv', index = False, header = True)
gexp_part = gexp[100:150]
gexp_part.to_csv('nolle_gexp13_batch3_part53.csv', index = False, header = True)
gexp_part = gexp[150:200]
gexp_part.to_csv('nolle_gexp13_batch3_part54.csv', index = False, header = True)
gexp_part = gexp[200:]
gexp_part.to_csv('nolle_gexp13_batch3_part55.csv', index = False, header = True)

gexp = pd.read_csv('results_ateriores/gexp14_all_res_batch3_aalst.csv')
gexp = gexp[gexp['batch_nro'].isin([3])]
gexp_part = gexp[:10]
gexp_part.to_csv('aalst_gexp14_batch3_part51.csv', index = False, header = True)
gexp_part = gexp[10:20]
gexp_part.to_csv('aalst_gexp14_batch3_part52.csv', index = False, header = True)
gexp_part = gexp[20:30]
gexp_part.to_csv('aalst_gexp14_batch3_part53.csv', index = False, header = True)
gexp_part = gexp[30:40]
gexp_part.to_csv('aalst_gexp14_batch3_part54.csv', index = False, header = True)
gexp_part = gexp[40:]
gexp_part.to_csv('aalst_gexp14_batch3_part55.csv', index = False, header = True)


