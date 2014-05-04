#!/usr/bin/python
# Import required libraries
import sys

# Start a counter and store the textfile in memory
count = 0
total_age = 0
count_lines = 0
max_age = 0
lines = sys.stdin.readlines()
lines.pop(0)

# For each line, find the sum of index 2 in the list.
for line in lines:
  count = count + int(line.strip().split(',')[2])
  total_age = total_age + int(line.strip().split(',')[0])
  if max_age < int(line.strip().split(',')[0]):
  	max_age = int(line.strip().split(',')[0])
  else:
  	max_age = max_age
  count_lines = count_lines + 1 
avg_age = total_age/count_lines
avg_impresion = count/count_lines

print 'Total Unique Visitors: ', count
print 'Average Age: ', avg_age
print 'Max Age: ', max_age
print 'Average Clicks per Impressions: ', avg_impresion

### EOF ###