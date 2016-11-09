
import re
import csv
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

pattern = re.compile(r"(\d+) (.+), (.+), CA (\d+), USA")

class Employee(object):
    def __init__(self,segments):
        # address
        matched = pattern.match(segments[0])
        if matched is None:
            raise Exception('format not supported')

        self.building_no = int( matched.group(1) )
        self.street = matched.group(2).lower()
        self.city = matched.group(3)
        self.zipcode = int( matched.group(4) )

        # employee-id
        self.id = int(segments[1])

employees = []
invalid_employees = []

address_file = "Employee_Addresses.csv"
with open(address_file,"rt") as inf:
    reader = csv.reader(inf)
    for segments in reader:
        try:
            employees.append(Employee(segments))
        except:
            invalid_employees.append(segments[0])

################
streets_counter = Counter((e.street for e in employees))
streets_counts = pd.Series(streets_counter)
streets_counts.sort_values(ascending=False)








