
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f'EarlyStopping: Initializing best loss to {val_loss:.6f}')
            return False
        
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: Loss did not improve from {self.best_loss:.6f}. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'EarlyStopping: Early stopping triggered after {self.patience} epochs without improvement')
                return True
        else:
            if self.verbose:
                print(f'EarlyStopping: Loss improved from {self.best_loss:.6f} to {val_loss:.6f}. Resetting counter.')
            self.best_loss = val_loss
            self.counter = 0
            
        return False
