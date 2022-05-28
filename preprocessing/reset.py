import os

numpy_path = r'memory\amt_in_buckets.npy'
processed_files_path = r'memory\processed_files.npy'
output_csv = r'data\csv_output\params.csv'
processed_rows_path = r'memory\processed_rows.npy'

print('***********************************')
print('************ WARNING **************')
print('***********************************')
print('** YOU ARE ABOUT TO CLEAR MEMORY **')
print('***********************************')
print('')
print('Do you wish to continue? This will reset the memory files and the outputs')
answer = input('Press enter to continue, exit the program to cancel deletion.')

if os.path.exists(processed_files_path):
    os.remove(processed_files_path)

if os.path.exists(numpy_path):
    os.remove(numpy_path)

if os.path.exists(output_csv):
    os.remove(output_csv)

if os.path.exists(processed_rows_path):
    os.remove(processed_rows_path)

for file in os.listdir(r'data\img_output'):
    os.remove(r'data\img_output' + '\\' + file)

print('')
print('Memory deleted')
