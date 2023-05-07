# Projet programmation graphique CUDA

- BERTHELOT Thomas
- VACHER Antoine
- CAVILLON Simon


### Filtres implémentés

|**Filtre**|**Séquentiel**|**CUDA**|**CUDA-Optimisé SharedMemory**|**CUDA-Optimisé Streams**|
|-------|-----------|---------|--------|--------|
|Edge| ✅ | ✅ | ✅ | ✅ |
|Gaussian Blur | ✅ | ✅ | ❌ *(Illegal mem. access)* | ✅ |
|Sobel | ✅ | ✅ | ✅ | ✅ |
|Laplacian | ✅ | ✅ | ❌ *(Illegal mem. access)* | ✅ |

#### Spécifications de la machine de test
```
CPU:
Model name:            AMD Ryzen 7 5800X 8-Core Processor
    CPU family:          25
    Model:               33
    Thread(s) per core:  2
    Core(s) per socket:  8
    Socket(s):           1
    CPU max MHz:         4850,1948
    CPU min MHz:         2200,0000

GPUs:
Number of devices: 1
Device Number: 0
    Device name: NVIDIA GeForce RTX 3060 Ti
    Memory Clock Rate (MHz): 6836
    Memory Bus Width (bits): 256
    Peak Memory Bandwidth (GB/s): 448.1
    Total global memory (Gbytes) 7.8
    Shared memory per block (Kbytes) 48.0
    minor-major: 6-8
    Warp-size: 32
    Concurrent kernels: yes
    Concurrent computation/communication: yes
    Max threads / block: 1024
    Max block dimension: 1024,1024,64
```

Lien vers les données récupéré en temps d'execution : [Google Sheets](https://docs.google.com/spreadsheets/d/1bn7UBBme0NnAXw3YTcCiOHWbbm8P4_v_us5Q3tGkuUI/edit?usp=sharing).
