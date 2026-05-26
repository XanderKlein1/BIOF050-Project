import tensorflow as tf
import gc
import tracemalloc


    
class MemoryLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_every_n_batches=50):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.snapshots = {}
        
    def on_train_begin(self, logs=None):
        tracemalloc.start()
        gc.collect()
        self.snapshots['train_start'] = tracemalloc.take_snapshot()
       
       
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        snapshot = tracemalloc.take_snapshot()
        self.snapshots[f'epoch_{epoch}_end'] = snapshot
        
        if epoch > 0:
            prev_snapshot = self.snapshots[f'epoch_{epoch-1}_end']
            top_stats = snapshot.compare_to(prev_snapshot, 'lineno')
            
            print(f"\n{'='*60}")
            print(f"Memory growth from Epoch {epoch-1} to Epoch {epoch}:")
            print(f"{'='*60}")
            for stat in top_stats[:10]:
                print(stat)
            print(f"{'='*60}\n")
        
class ManageQueue(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_end(self, logs=None):
        self.model.queue = tf.zeros_like(self.model.queue)