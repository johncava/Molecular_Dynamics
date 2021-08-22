#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


def readPDB(fName, atom=True):
    f = open(fName)
    l = f.readlines()
    f.close()
    
    fNameTemp = fName.rstrip("*.pdb") + "_temp.pdb"
    fTemp = open(fNameTemp,'w')

    for line in l:
        lineSplit = line.split()
        if len(lineSplit):
            firstWord = line.split()[0]
        else:
            firstWord = "None"

        if atom:
            if firstWord == "ATOM":
                newLine = line[12:16] + " " + line[17:20] + " " + line[21] + " " + line[22:26] + " " + line[30:38] + " " + line[38:46] + " " + line[46:54] + " " + line[54:60] + " " + line[60:66] + " " + line[72:76] + " " + line[76:78] + "\n"
                fTemp.write(newLine)
                
        else:
            if (firstWord == "ATOM") or (firstWord == "HETATM") or (firstWord == "ANISOU"):
                newLine = line[12:16] + " " + line[17:20] + " " + line[21] + " " + line[22:26] + " " + line[30:38] + " " + line[38:46] + " " + line[46:54] + " " + line[54:60] + " " + line[60:66] + " " + line[72:76] + " " + line[76:78] + "\n"
                fTemp.write(newLine)

    fTemp.close()
    
    #os.remove(fNameTemp)
    #df = pd.read_csv(fNameTemp, sep=r"\s+", names=['AtomName', 'Resname', 'Chain', 'Resid', 'X', 'Y', 'Z', 'Occu', 'Beta', 'Segname', 'Symbol'])
    df = pd.read_csv(fNameTemp, sep=r"\s+", names=['AtomName', 'Resname', 'Chain', 'Resid', 'X', 'Y', 'Z', 'Occu', 'Beta', 'Symbol'])
    if df['Symbol'].isna().all():
        df = df.drop(['Symbol'], axis=1)

    return df


# In[3]:


def readPSF(fName):
    f = file(fName,"r")
    l = f.readlines()
    f.close()
    for i in range(len(l)):
        thisLine = l[i]
        if "NATOM" in thisLine:
            start = i+1
        if "NBOND" in thisLine:
            stop = i-1
            break
    names = ["Segname", "Resid", "Resname", "AtomType", "AtomName", "Charge","Mass", "Temp"]
    df = pd.read_table(fName, header=None, names=names, delimiter=r"\s+", skiprows=start,nrows=(stop-start))
    df2 = df.drop(["Temp"], axis=1)
    return df2


# In[4]:


def writePDB(obj, fName):
    f = open(fName, 'w')
    count = 1
    for i in range(0,obj.shape[0]):
        testLine = obj.iloc[i]
        printLine = "ATOM  "
        printLine += str(count).rjust(5)
        count += 1
        if len(testLine.AtomName) <= 3:
            printLine += "  " + testLine.AtomName.ljust(4)
        else:
            printLine += " " + testLine.AtomName.ljust(4)
            printLine = printLine.ljust(17)

        printLine += testLine.Resname.rjust(3)
        printLine += " " + testLine.Chain
        printLine += str(testLine.Resid).rjust(4)
        printLine += "    " + str('%.3f'%testLine.X).rjust(8)
        printLine += str('%.3f'%testLine.Y).rjust(8)
        printLine += str('%.3f'%testLine.Z).rjust(8)
        printLine += str('%.2f'%testLine.Occu).rjust(6)
        printLine += str('%.2f'%testLine.Beta).rjust(6)
        #printLine += "     " + testLine.Segname.rjust(4)
        
        #print testLine.index
        #if isinstance(testLine.Symbol, str):
        if 'Symbol' in testLine.index:
            if isinstance(testLine.Symbol, str):
                printLine += " " + testLine.Symbol.rjust(2)
            else:
                printLine += "   "
        else:
            printLine += "   "
            
        printLine += "\n"
        f.write(printLine)
    f.write("END\n")
    f.close()


# In[ ]:




