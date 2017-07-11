output_file_path = 'submission.txt'
list_file_path = 'data/list_val.txt'

fd = open(list_file_path)
groundtruth = {line.split()[0].split('.')[0] : line.split()[1] for line in fd}
fd.close()

fd = open(output_file_path)
prediction = {line.split()[1] : line.split()[0] for line in fd}
fd.close()

correct = 0
for idx in groundtruth.keys():
    if idx in prediction and groundtruth[idx] == prediction[idx]:
        correct += 1
    elif idx not in prediction:
        print(idx)

tot = len(groundtruth.keys())
print('{} / {} = {}'.format(correct, tot, correct * 1.0 / tot))
