# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 00:36:44 2020

@author: rbrow
"""

from bs4 import BeautifulSoup
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from os import path


def tagortext (x):
    try:
        out = x.contents
        out = ""
    except AttributeError:
        out = x   
    return(out)


def parsebody (body):
    if(len(body)==0):
        out = "👍" # replace with the thumbs up emoji
    elif(len(body)==1):        
        out = tagortext(body[0])
    else:
        allbodyelems = [tagortext(i) for i in body]
        out = ' '.join(allbodyelems)
    
    return(out)        
    

def fbhtml_to_df (save=False):
    root = tk.Tk()
    root.withdraw()
    
    # dialog to choose path
    file_path = filedialog.askopenfilename()
    dirnm = path.dirname(file_path)
    filepref = path.splitext(file_path)[0]
    
    # implement checks for correct file type
        
    # open and parse with beatufiul soup
    page = open(file_path,encoding='utf-8').read()
    fbsoup = BeautifulSoup(page,features="lxml")
    
    # could be used to throw errors if not a FB html
    # tst = fbsoup.find_all("div",class_="_2t-a _26aw _5rmj _50ti _2s1")      
    
    
    # uses the div class value to identify all messages
    # msgdivs = fbsoup.find_all("div", class_="pam _3-95 _2pi0 _2lej uiBoxWhite noborder")
    namedivs = fbsoup.find_all("div",class_="_3-96 _2pio _2lek _2lel")
    datedivs = fbsoup.find_all("div",class_="_3-94 _2lem")
    txtdivs = fbsoup.find_all("div",class_="_3-96 _2let")
    
    if(len(datedivs) > len(txtdivs)): # group conversations have an extra date div at the beginning
        datedivs = datedivs[1:]
    
    # grab the name and date-timestamp for each message
    users = [i.contents[0] for i in namedivs]
    dates = [i.contents[0] for i in datedivs]
    txts = [parsebody(i.div.contents[1].contents) for i in txtdivs]    
    
    # save to dataframe and export
    outdf = pd.DataFrame({'user':users,'datetimestr':dates,'body':txts})
    
    if(save):
        outdf.to_csv(path.join(dirnm,filepref) +'.csv',encoding='utf-8',index=False)
        
        
    return({'df':outdf,'filepref':filepref,'dirpath':dirnm})


if __name__ == '__main__':
    fbhtml_to_df(save=True)