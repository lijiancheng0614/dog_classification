phase = 'val'

fd = open('data_{}.txt'.format(phase))
lines = [line.split()[:2] for line in fd]
fd.close()

fd = open('label_name.txt')
label_name = [line.split()[0].split('---')[-1] for line in fd]
fd.close()

image_set = dict()
for line in lines:
    idx, label = line
    label = int(label)
    if idx in image_set.keys():
        print('{} {} {} {} {}'.format(idx,
            image_set[idx], label_name[image_set[idx]],
            label, label_name[label]))
        image_set[idx] = None
        continue
    image_set[idx] = label

label_id = set()
fd = open('list_{}.txt'.format(phase), 'w')
for idx in image_set.keys():
    label = image_set[idx]
    if label is None:
        continue
    fd.write('{}.jpg {}\n'.format(idx, label))
    if label not in label_id:
        label_id.add(label)

fd.close()

print(len(label_id))
print(label_id)
