import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


def test(model, instances, labels):
    crit = torch.nn.BCELoss(reduction='mean')
    eta = model(instances)
    p = torch.sigmoid(eta)
    fpr, tpr, _ = roc_curve(labels, eta.detach().numpy())
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc(fpr, tpr)})' )
    plt.legend(loc="lower right");plt.show()
    print("Loss: ", float(crit(p, labels)))
    print(confusion_matrix((p>0.5).numpy(),(labels>0.5).numpy()))
    return p>0.5
    
