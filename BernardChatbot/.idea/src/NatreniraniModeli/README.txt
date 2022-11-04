------------------------------------------------
--------------------model.h5--------------------
------------------------------------------------

# Maksimalna duljina rečenice
MAX_LENGTH = 40

# Maksimalni broj sample-ova za preprocesirati
MAX_SAMPLES = 250000

# Za dataset
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
BUFFER_SIZE = 20000

# Parametri za transformer
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

# Broj epoha
EPOCHS = 40

------------------------------------------------
-------------------model2.h5-------------------- 
------------------------------------------------

# Maksimalna duljina rečenice
MAX_LENGTH = 40

# Maksimalni broj sample-ova za preprocesirati
MAX_SAMPLES = 250000

# Za dataset
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
BUFFER_SIZE = 20000

# Parametri za transformer
NUM_LAYERS = 4
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

# Broj epoha
EPOCHS = 40

------------------------------------------------
-------------------model3.h5--------------------
------------------------------------------------

# Maksimalna duljina rečenice
MAX_LENGTH = 40

# Maksimalni broj sample-ova za preprocesirati
MAX_SAMPLES = 250000

# Za dataset
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
BUFFER_SIZE = 20000

# Parametri za transformer
NUM_LAYERS = 2
D_MODEL = 512
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

# Broj epoha
EPOCHS = 40

------------------------------------------------
----------model_malo_podataka.h5----------------
------------------------------------------------


# Maksimalna duljina rečenice
MAX_LENGTH = 40

# Maksimalni broj sample-ova za preprocesirati
MAX_SAMPLES = 50000

# Za dataset
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
BUFFER_SIZE = 20000

# Parametri za transformer
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

# Broj epoha
EPOCHS = 40