def read_labels():
    labels = []
    with open('train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels.append(row[1])
    return labels

labels = read_labels()
classes = set(labels)

with open('classes.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(list(classes))
