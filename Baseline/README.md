# Nutzung der Baseline

## Python Packages
Zuerst sollte sichergestellt werden, dass alle nötigen Packages installiert sind.
Installiert sein müssen: 
* Python
* Numpy
* TensorFlow
* Sklearn
* Natsort
* Matplotlib
* Pandas 
* PIL

## Starten der *.py-Datei
Diese wird zum Training verwendet

Für das Training sind die Trainingsdaten nötig. Diese müssen erst geniert werden. Eine Anleitung dafür kann auf dem Git-Repository von [Melesse](https://github.com/micmelesse/3D-reconstruction-with-Neural-Networks) gefunden werden.
Die Pfad zu den Trainingsdaten muss über den Aufruf stattfinden.


Anschließend sollte die GPU im Code festgelegt werden, welche genutzt werden soll. Im vorliegenden Fall wurde die GPU 4 verwendet.

```
os.environ["CUDA_VISIBLE_DEVICES"]="4"
```

Das Training wird gestartet durch folgenden Aufruf in der Konsole gestartet:
```
python3 pipeline.py --data_preprocessed="data_preprocessed" --log_path="logs/fit/" --epochs=10 --batchsize=2
```

Dabei wird bei data_preprocessed der Pfad zu den Traininsdaten angegeben, log_path der Pfad in dem die Logs gespeichert werden, epcohs für die Epochenanzahl und Batchsize für die Anzahl der Batchs * 24 (durch die 24 Ansichten der Bilder)


Nach Beendung des Training wurde eine *.hd5f-Datei erstellt. Mit dieser kann das neuronale Netz getestet werden.

## Inferenz 
Diese wird über das Jupyter Notebook ausgeführt.
In den ersten Abschnitten ist ein ähnlicher Code wie in der *.py-Datei zu sehen. Das Notebook wrude für die Entwicklung des neuronalen Netzes genutzt. Im unteren Teil sind die Abschnitte zu sehen, welche für die Inferenz zuständig sind. Dabei müssen die Checkpoints angegeben werden und die Pfade zu den Bildern, zu welchem ein 3D-Modell erstellt werden soll. Zur Visualisierung wird ein weiteres Package "matplotlib" genutzt.
Außerdem ist hier die Umwandlung des Modells in ein mobil-fähiges Modell zu sehen.

