# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 16:11:57 2016

@author: bhavesh
"""


#I = ([])
#for k in range(21):
#    I.append(quad(f,0,1,(k)))
#
#I = np.array(I)
#I = I[:,0]


#workbook = xlsxwriter.Workbook('Ik.xlsx')
#worksheet = workbook.add_worksheet()
#col = 0
#for col,data in enumerate(I):
#    worksheet.write_column(col,float(data))
#workbook.close()

#print("\n{}".format(I))
#
##Check for the recurrence relation: 
#I_ch = [1-np.e**(-1)]
##Ik=1-np.e**(-1)
#for k in range(20):
#    Ik = 1-(k+1)*I_ch[k]    
#    I_ch.append(Ik)
#
#I_ch = np.array(I_ch)

#print("Difference: {}".format(I_ch-I))

# Writing the Ouput to the excel file to get the values: 

import xlsxwriter