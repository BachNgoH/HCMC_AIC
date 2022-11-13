from rapidfuzz import fuzz

import pandas as pd 
import json
import re 
import numpy as np 

def fill_ocr(st, list_ocrs):
  def check(x):
    score = fuzz.token_set_ratio(x.lower(), st.lower())
    if score >= 80:
      if re.search(f'\s{st.lower()}\s',x.lower()):
        return True
      return False
    return False
  
  temp_ocr = list(filter(check, list_ocrs))
  list_ocr=list(map(lambda x:f'{x.split(",")[0]}/{x.split(",")[1]:0>6}.jpg', temp_ocr))

  print(list_ocr)

  return list_ocr

if __name__ == "__main__":
    with open("/content/C01_V02-03_C02_V04.txt", "r", encoding="utf8") as fi:
        list_ocrs = list(map(lambda x: x.replace("\n",""), fi.readlines()))
    fill_ocr("quốc thanh", list_ocrs)
    print(fill_ocr)
