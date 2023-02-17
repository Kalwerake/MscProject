import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

class MetricShow:

    def __init__(self, path ):
        self.path = path
        with open(path, 'rb') as h:
            self.metric_dict = pickle.load(h)

        self.tr_epochs = len(self.metric_dict['train_loss_history'])
        self.te_epochs = len(self.metric_dict['test_loss'])

    def show_best(self, name, tell = True):
        test_acc = list(self.metric_dict[name])
        max_val_acc = max(test_acc)
        ep = test_acc.index(max_val_acc)
        if tell:
            print(f'maximum validation accuracy: {max_val_acc}, at epoch: {ep + 1}')
        return (ep, max_val_acc)

    def plot_metrics(self, model_name, save = False):
        x_tr = range(self.tr_epochs)
        x_te = range(self.te_epochs)
        fig, (ax0,ax1)= plt.subplots(nrows = 1, ncols=2, figsize = (10,5))
        fig.suptitle(model_name)
        ax0.plot(x_tr,self.metric_dict['train_loss_history'], color = 'orange', label= 'training')
        ax0.plot(x_te,self.metric_dict['test_loss'], color = 'blue', label= 'validation')
        ax0.legend(loc='upper left')
        ax0.set_title('Loss')
        ax0.set_xlabel('Epoch')
        ax0.set_ylabel('Loss')

        ax1.plot(x_tr, self.metric_dict['train_acc_history'], color = 'orange', label = 'training')
        ax1.plot(x_te, self.metric_dict['test_acc'], color = 'blue', label = 'validation')
        ax1.legend(loc='upper left')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        if save:
            save_path = os.path.join(os.getcwd(), 'model_evaluation' , f'{model_name}.png')
            plt.savefig(save_path)

        plt.show()

    def plot_cm(self,model_name):
        cm = confusion_matrix(self.metric_dict['y_true'], self.metric_dict['y_pred'])


        plt.figure(figsize=(8, 6))
        # Create heatmap
        sns.heatmap(cm, annot=True, cbar=None, fmt="d")

        plt.title(model_name), plt.tight_layout()

        plt.ylabel("True Class"),
        plt.xlabel("Predicted Class")
        plt.show()