import os
import csv

indir = './ClipSets/'
outdir= './ClipSets_new/'

actions_autotrain = {}
actions_train = {}
actions_test = {}

with open(indir+'actions_autotrain.txt', 'r') as autotrain:
    for line in autotrain.readlines():
        actions_autotrain[line[:-4]] = {}

with open(indir+'actions_test.txt', 'r') as test:
    for line in test.readlines():
        actions_test[line[:-4]] = {}

with open(indir+'actions_train.txt', 'r') as train:
    for line in train.readlines():
        actions_train[line[:-4]] = {}

for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        if f in ['actions_autotrain.txt', 'actions_test.txt', 'actions_train.txt']:
            continue
        action_name, file_type = f.split("_")
        if file_type == "autotrain.txt":
            with open(indir+f, 'r') as file:
                for line in file.readlines():
                    filename, label = line.split("  ")
                    actions_autotrain[filename][action_name] = 1 if label[:-1] == '1' else -1
        elif file_type == "train.txt":
            with open(indir+f, 'r') as file:
                for line in file.readlines():
                    filename, label = line.split("  ")
                    actions_train[filename][action_name] = 1 if label[:-1] == '1' else -1
        else:
            with open(indir + f, 'r') as file:
                for line in file.readlines():
                    filename, label = line.split("  ")
                    actions_test[filename][action_name] = 1 if label[:-1] == '1' else -1

action_names = ['HandShake', 'DriveCar', 'FightPerson', 'Eat', 'StandUp', 'SitUp', 'Kiss',
                'GetOutCar', 'Run', 'HugPerson', 'AnswerPhone', 'SitDown']

with open('actions_autotrain.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['file_name']+action_names)
    for file_name in list(actions_autotrain.keys()):
        writer.writerow([file_name]+[actions_autotrain[file_name][key] for key in action_names])

with open('actions_train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['file_name']+action_names)
    for file_name in list(actions_train.keys()):
        writer.writerow([file_name]+[actions_train[file_name][key] for key in action_names])
        
with open('actions_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['file_name']+action_names)
    for file_name in list(actions_test.keys()):
        writer.writerow([file_name]+[actions_test[file_name][key] for key in action_names])