# Nutzung des IM-NETs

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
* OpenCV-Python
* PyMCubes
* MobileNetV2 von TF-Slim

## Starten des Projekts
Dieses Projekt basiert auf dem Code der originalen Implementierung von IM-NET. Es wurde der Encoder verändert, in dem MobileNetV2 implementiert wurde. Dazu ist zwingend das MobileNetV2 Package von TF-Slim bzw. Tensorflow nötig. Zum Training werden die gerenderten Views von 3D-R2n2, Voxel-Daten von HSO nd die Punktkoordinaten von IM-NET benötigt. All diese könne nauf der [Github-Seite](https://github.com/czq142857/IM-NET) gefunden werden. 
Gestartet werden kann das Projekt über die Kommandozeile. Dabei können verschiedene Parameter angegeben werden. Diese können an den folgenden Befehl angehängt werden. Für die Infernz wird das Training als "False" beschriftet und zu allen Bildern aus dem "image"-Ordner wird ein 3D-Modell erzeugt.

```
pyhton3 main.py
```


